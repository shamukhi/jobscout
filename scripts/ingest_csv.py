"""Load a real jobs CSV (e.g. a Kaggle LinkedIn/Indeed postings dump) into the running app or a local index.

  python scripts/ingest_csv.py jobs.csv --title-col job_title --company-col company --location-col location \
      --desc-col job_description --skills-col job_skills --date-col first_seen --url-col job_link \
      --post http://localhost:8000 --admin-token $ADMIN_TOKEN [--prune]

Job type / seniority / remote are inferred from keywords because most public dumps lack clean columns.
Check a sample of the output before trusting it.
"""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from urllib import request


def infer(title: str, desc: str):
    t, d = title.lower(), desc.lower()
    if "intern" in t:
        job_type, seniority = "internship", "intern"
    else:
        job_type = "contract" if "contract" in t else "part-time" if "part-time" in t or "part time" in t else "full-time"
        seniority = "senior" if any(w in t for w in ("senior", "sr.", "staff", "lead", "principal")) else \
                    "junior" if any(w in t for w in ("junior", "jr", "associate", "graduate", "entry")) else "mid"
    remote = any(w in t or w in d[:600] for w in ("remote", "work from home", "wfh"))
    return job_type, seniority, remote


def parse_date(s: str) -> str:
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            pass
    return datetime.now(timezone.utc).isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--title-col", default="title")
    ap.add_argument("--company-col", default="company")
    ap.add_argument("--location-col", default="location")
    ap.add_argument("--desc-col", default="description")
    ap.add_argument("--skills-col", default=None, help="comma/pipe separated; omit if absent")
    ap.add_argument("--date-col", default=None)
    ap.add_argument("--url-col", default=None)
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--out", default="jobs_ingest.json")
    ap.add_argument("--post", default=None, help="base URL of a running Jobscout to POST /ingest to")
    ap.add_argument("--admin-token", default=None)
    ap.add_argument("--prune", action="store_true")
    a = ap.parse_args()

    jobs, seen = [], set()
    with open(a.csv, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            title, desc = (row.get(a.title_col) or "").strip(), (row.get(a.desc_col) or "").strip()
            if not title or not desc:
                continue
            company, loc = (row.get(a.company_col) or "Unknown").strip(), (row.get(a.location_col) or "Unknown").strip()
            jid = hashlib.sha1(f"{title}|{company}|{loc}".encode()).hexdigest()[:12]  # stable across re-ingests
            if jid in seen:
                continue
            seen.add(jid)
            job_type, seniority, remote = infer(title, desc)
            skills = [s.strip() for s in (row.get(a.skills_col) or "").replace("|", ",").split(",") if s.strip()] if a.skills_col else []
            jobs.append({
                "id": jid, "title": title, "company": company, "location": loc, "remote": remote,
                "job_type": job_type, "seniority": seniority, "description": desc[:4000], "skills": skills[:15],
                "posted_at": parse_date(row.get(a.date_col) or "") if a.date_col else datetime.now(timezone.utc).isoformat(),
                "url": (row.get(a.url_col) or None) if a.url_col else None,
            })
            if len(jobs) >= a.limit:
                break

    with open(a.out, "w") as f:
        json.dump(jobs, f)
    print(f"parsed {len(jobs)} jobs -> {a.out}")
    if a.post:
        req = request.Request(a.post.rstrip("/") + "/ingest", data=json.dumps({"jobs": jobs, "prune": a.prune}).encode(),
                              headers={"Content-Type": "application/json", **({"X-Admin-Token": a.admin_token} if a.admin_token else {})})
        print(request.urlopen(req).read().decode())


if __name__ == "__main__":
    sys.exit(main())
