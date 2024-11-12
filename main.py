from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from sqlalchemy import create_engine, Column, String, DateTime, Integer, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.orm import declarative_base
from redis import Redis
from datetime import datetime
import string
import random
import validators
from typing import Optional, List
from pydantic import BaseModel
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
import os
from dotenv import load_dotenv


# 데이터베이스 설정
load_dotenv() # .env 파일 로드
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Redis 설정
redis = Redis(host='localhost', port=6379, db=0)


# 모델 정의
class URL(Base):
    __tablename__ = 'urls'

    short_id = Column(String(50), primary_key=True)
    original_url = Column(String(2048), nullable=False)
    custom_path = Column(String(50), unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(100), nullable=True)
    visits = Column(Integer, default=0)
    last_visited = Column(DateTime, nullable=True)


class URLStats(Base):
    __tablename__ = 'url_stats'

    id = Column(Integer, primary_key=True)
    short_id = Column(String(50))
    visited_at = Column(DateTime, default=datetime.utcnow)
    referrer = Column(String(2048), nullable=True)
    user_agent = Column(String(1024), nullable=True)
    ip_address = Column(String(45), nullable=True)


# Pydantic 모델
class URLCreate(BaseModel):
    url: str
    custom_path: Optional[str] = None
    created_by: Optional[str] = None


class URLResponse(BaseModel):
    short_id: str
    original_url: str
    custom_path: Optional[str]
    created_at: datetime
    visits: int


class URLStats(BaseModel):
    total_visits: int
    visits_by_day: List[dict]
    top_referrers: List[dict]


# FastAPI 앱 설정
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 프론트엔드 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# URL 생성 및 관리 엔드포인트
@app.post("/api/urls", response_model=URLResponse)
async def create_url(url_data: URLCreate, db: Session = Depends(get_db)):
    if not validators.url(url_data.url):
        raise HTTPException(status_code=400, detail="Invalid URL")

    # 커스텀 경로 검증
    if url_data.custom_path:
        if len(url_data.custom_path) < 4:
            raise HTTPException(status_code=400, detail="Custom path too short")
        existing = db.query(URL).filter_by(custom_path=url_data.custom_path).first()
        if existing:
            raise HTTPException(status_code=400, detail="Custom path already taken")
        short_id = url_data.custom_path
    else:
        # 랜덤 short_id 생성
        while True:
            short_id = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
            if not db.query(URL).filter_by(short_id=short_id).first():
                break

    url_db = URL(
        short_id=short_id,
        original_url=url_data.url,
        custom_path=url_data.custom_path,
        created_by=url_data.created_by
    )
    db.add(url_db)
    db.commit()
    db.refresh(url_db)

    # Redis 캐시에 저장
    redis.setex(f"url:{short_id}", 3600, url_data.url)

    return url_db


@app.get("/api/stats/{short_id}", response_model=URLStats)
@cache(expire=300)
async def get_url_stats(short_id: str, db: Session = Depends(get_db)):
    url = db.query(URL).filter_by(short_id=short_id).first()
    if not url:
        raise HTTPException(status_code=404, detail="URL not found")

    # 통계 데이터 수집
    stats = db.query(URLStats).filter_by(short_id=short_id)

    # 일별 방문 통계
    daily_visits = db.query(
        func.date(URLStats.visited_at).label('date'),
        func.count().label('visits')
    ).filter_by(short_id=short_id).group_by(func.date(URLStats.visited_at)).all()

    # 상위 레퍼러
    top_referrers = db.query(
        URLStats.referrer,
        func.count().label('count')
    ).filter_by(short_id=short_id).group_by(URLStats.referrer).limit(5).all()

    return {
        "total_visits": url.visits,
        "visits_by_day": [{"date": str(d.date), "visits": d.visits} for d in daily_visits],
        "top_referrers": [{"referrer": r.referrer or "Direct", "count": r.count} for r in top_referrers]
    }


@app.get("/{short_id}")
async def redirect_url(
        short_id: str,
        background_tasks: BackgroundTasks,
        request: Request,
        db: Session = Depends(get_db)
):
    # Redis 캐시에서 먼저 확인
    cached_url = redis.get(f"url:{short_id}")
    if cached_url:
        original_url = cached_url.decode('utf-8')
    else:
        url = db.query(URL).filter_by(short_id=short_id).first()
        if not url:
            raise HTTPException(status_code=404, detail="URL not found")
        original_url = url.original_url
        # 캐시에 저장
        redis.setex(f"url:{short_id}", 3600, original_url)

    # 비동기로 통계 업데이트
    background_tasks.add_task(update_stats, short_id, request, db)

    return RedirectResponse(original_url)


async def update_stats(short_id: str, request: Request, db: Session):
    url = db.query(URL).filter_by(short_id=short_id).first()
    url.visits += 1
    url.last_visited = datetime.utcnow()

    # 방문 통계 저장
    stats = URLStats(
        short_id=short_id,
        referrer=request.headers.get('referer'),
        user_agent=request.headers.get('user-agent'),
        ip_address=request.client.host
    )
    db.add(stats)
    db.commit()


# 관리자 엔드포인트
@app.get("/api/admin/urls", response_model=List[URLResponse])
async def get_all_urls(
        skip: int = 0,
        limit: int = 10,
        db: Session = Depends(get_db)
):
    urls = db.query(URL).offset(skip).limit(limit).all()
    return urls


if __name__ == "__main__":
    import uvicorn

    Base.metadata.create_all(bind=engine)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)