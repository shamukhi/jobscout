from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .textutil import norm_skill

JobType = Literal["internship", "full-time", "part-time", "contract"]
Seniority = Literal["intern", "junior", "mid", "senior"]


class Job(BaseModel):
    id: str
    title: str
    company: str
    location: str
    remote: bool = False
    job_type: JobType = "full-time"
    seniority: Seniority = "junior"
    description: str
    skills: list[str] = Field(default_factory=list)
    posted_at: datetime
    salary_text: str | None = None
    url: str | None = None

    @field_validator("skills")
    @classmethod
    def _norm(cls, v: list[str]) -> list[str]:
        return sorted({norm_skill(s) for s in v if s.strip()})

    @field_validator("posted_at")
    @classmethod
    def _tz(cls, v: datetime) -> datetime:
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)

    def embed_text(self) -> str:
        where = f"{self.location}{', remote' if self.remote else ''}"
        return (
            f"{self.title} at {self.company} ({where}). "
            f"{self.job_type} role, {self.seniority} level. "
            f"Skills: {', '.join(self.skills)}. {self.description}"
        )

    def embed_hash(self) -> str:
        return hashlib.sha256(self.embed_text().encode()).hexdigest()[:16]

    def full_hash(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]

    def age_days(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        return max((now - self.posted_at).total_seconds() / 86400, 0.0)


class SearchFilters(BaseModel):
    location: str | None = None
    remote: bool | None = None
    job_type: JobType | None = None
    seniority: Seniority | None = None
    max_age_days: int | None = None

    def matches(self, job: Job) -> bool:
        if self.remote is not None and job.remote != self.remote:
            return False
        if self.job_type and job.job_type != self.job_type:
            return False
        if self.seniority and job.seniority != self.seniority:
            return False
        if self.location and self.location.lower() not in job.location.lower():
            return False
        if self.max_age_days is not None and job.age_days() > self.max_age_days:
            return False
        return True


class Profile(BaseModel):
    skills: list[str] = Field(default_factory=list)
    target_roles: list[str] = Field(default_factory=list)
    preferred_locations: list[str] = Field(default_factory=list)
    remote_preference: Literal["remote", "onsite", "any"] = "any"
    seniority: Seniority | None = None
    job_types: list[JobType] = Field(default_factory=list)
    notes: str = ""

    @field_validator("skills")
    @classmethod
    def _norm(cls, v: list[str]) -> list[str]:
        return sorted({norm_skill(s) for s in v if s.strip()})

    def merge(self, patch: dict) -> "Profile":
        data = self.model_dump()
        for key, value in patch.items():
            if value is None or key not in data:
                continue
            if isinstance(data[key], list):
                data[key] = list(dict.fromkeys([*data[key], *value]))  # union, keep order
            else:
                data[key] = value
        return Profile(**data)

    def is_empty(self) -> bool:
        return self == Profile()


class SearchHit(BaseModel):
    job: Job
    score: float
    breakdown: dict[str, float]
    reasons: list[str]
    saved: bool = False
