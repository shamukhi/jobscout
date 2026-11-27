"""Generate a synthetic jobs feed (data/jobs.json) and an eval set (evals/queries.jsonl).

Relevance labels come from the generator's job families, so they are exact for THIS data.
That makes the eval a pipeline check, not a benchmark. For numbers worth quoting, ingest a real
dataset (see README) and hand-label ~30 queries.
"""
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rng = random.Random(7)
NOW = datetime.now(timezone.utc)

FAMILIES = {
    "llm": dict(
        titles=["LLM Engineer", "Generative AI Engineer", "Applied AI Engineer", "AI Agents Engineer"],
        skills=["python", "llm", "rag", "embeddings", "prompt engineering", "fastapi", "vector databases"],
        blurbs=[
            "Build chat assistants and retrieval-augmented generation pipelines on top of large language models.",
            "Design tool-using AI agents, evaluate model outputs and iterate on prompts and guardrails.",
            "Own semantic retrieval, prompt design and structured-output validation for our generative AI features.",
        ],
    ),
    "backend_py": dict(
        titles=["Backend Engineer", "Python Backend Developer", "API Engineer", "Software Engineer, Platform"],
        skills=["python", "fastapi", "postgresql", "redis", "docker", "rest apis", "sql"],
        blurbs=[
            "Build and scale Python services and REST APIs that power our product.",
            "Design reliable backend systems with caching, queues and clean database schemas.",
            "Ship production API endpoints, monitor them and keep latency low under load.",
        ],
    ),
    "backend_java": dict(
        titles=["Java Backend Engineer", "Software Engineer, Services"],
        skills=["java", "spring boot", "postgresql", "kafka", "microservices", "docker"],
        blurbs=[
            "Develop Spring Boot microservices that process high volumes of transactions.",
            "Build event-driven Java services on Kafka with strong consistency guarantees.",
        ],
    ),
    "data_eng": dict(
        titles=["Data Engineer", "Analytics Engineer", "ETL Developer"],
        skills=["python", "sql", "airflow", "spark", "dbt", "aws"],
        blurbs=[
            "Build batch and streaming data pipelines and keep the warehouse trustworthy.",
            "Model data with dbt, orchestrate workflows in Airflow and own data quality checks.",
        ],
    ),
    "ml": dict(
        titles=["Machine Learning Engineer", "Applied Scientist", "ML Platform Engineer"],
        skills=["python", "pytorch", "scikit-learn", "mlops", "docker", "aws"],
        blurbs=[
            "Train, evaluate and deploy machine learning models for production use cases.",
            "Build training pipelines and model serving infrastructure, and monitor model drift.",
        ],
    ),
    "search_rec": dict(
        titles=["Search Engineer", "Recommendations Engineer", "Relevance Engineer"],
        skills=["python", "elasticsearch", "embeddings", "ranking", "recommender systems", "sql"],
        blurbs=[
            "Improve search relevance and ranking with learning-to-rank and embedding retrieval.",
            "Build recommendation systems, run offline metrics and A/B tests to measure impact.",
        ],
    ),
    "frontend": dict(
        titles=["Frontend Engineer", "React Developer", "UI Engineer"],
        skills=["react", "typescript", "javascript", "css", "html"],
        blurbs=[
            "Build fast, accessible user interfaces in React and TypeScript.",
            "Turn designs into polished components and improve web performance.",
        ],
    ),
    "fullstack": dict(
        titles=["Full Stack Engineer", "Software Engineer, Web"],
        skills=["react", "node.js", "typescript", "postgresql", "rest apis"],
        blurbs=[
            "Own features end to end across a React frontend and a Node.js backend.",
            "Build web applications from database schema to user interface.",
        ],
    ),
    "analyst": dict(
        titles=["Data Analyst", "Business Intelligence Analyst"],
        skills=["sql", "power bi", "excel", "python", "tableau"],
        blurbs=[
            "Turn raw data into dashboards and reports that guide business decisions.",
            "Write SQL, build Power BI dashboards and present insights to stakeholders.",
        ],
    ),
    "devops": dict(
        titles=["DevOps Engineer", "Site Reliability Engineer", "Cloud Engineer"],
        skills=["docker", "kubernetes", "terraform", "aws", "ci/cd"],
        blurbs=[
            "Run infrastructure as code and keep deployments fast and reliable.",
            "Own Kubernetes clusters, CI/CD pipelines and incident response.",
        ],
    ),
}
COMPANIES = ["Nimbus Labs", "Kestrel AI", "Orbit Pay", "Lumen Health", "Quill Systems", "Fathom Data",
             "Brightpath", "Zephyr Cloud", "Alder & Co", "Cobalt Mobility", "Mosaic Learn", "Tidewater"]
CITIES = ["Bengaluru", "Gurugram", "Hyderabad", "Pune", "Mumbai", "Chennai"]
FILLERS = ["You will work in a small team and ship weekly.", "We value ownership and clear writing.",
           "Mentorship from senior engineers is part of the role.", "You will talk to users and own outcomes."]
PER_FAMILY = 12


def make_jobs():
    jobs, family_of = [], {}
    n = 0
    for fam, spec in FAMILIES.items():
        for _ in range(PER_FAMILY):
            n += 1
            seniority = rng.choices(["intern", "junior", "mid", "senior"], [3, 4, 3, 2])[0]
            job_type = "internship" if seniority == "intern" else rng.choices(["full-time", "contract"], [9, 1])[0]
            skills = rng.sample(spec["skills"], k=min(len(spec["skills"]), rng.randint(4, 5)))
            remote = rng.random() < 0.35
            jid = f"job-{n:03d}"
            jobs.append({
                "id": jid,
                "title": rng.choice(spec["titles"]),
                "company": rng.choice(COMPANIES),
                "location": rng.choice(CITIES),
                "remote": remote,
                "job_type": job_type,
                "seniority": seniority,
                "description": f"{rng.choice(spec['blurbs'])} {rng.choice(FILLERS)} Experience with {skills[0]} and {skills[1]} is expected.",
                "skills": skills,
                "posted_at": (NOW - timedelta(days=rng.randint(0, 60))).isoformat(),
                "salary_text": None,
                "url": f"https://example.com/jobs/{jid}",
            })
            family_of[jid] = fam
    return jobs, family_of


def main():
    jobs, family_of = make_jobs()
    (ROOT / "data" / "jobs.json").write_text(json.dumps(jobs, indent=1))

    def rel(families, **flt):
        out = []
        for j in jobs:
            if family_of[j["id"]] not in families:
                continue
            if "remote" in flt and j["remote"] != flt["remote"]:
                continue
            if "job_type" in flt and j["job_type"] != flt["job_type"]:
                continue
            out.append(j["id"])
        return out

    ai_profile = {"skills": ["python", "llm", "fastapi", "embeddings"], "job_types": ["internship"]}
    be_profile = {"skills": ["python", "fastapi", "redis", "postgresql"], "remote_preference": "remote"}
    queries = [
        ("build chatbots and retrieval augmented generation pipelines with language models", ["llm"], {}, None),
        ("python api developer with redis and postgres", ["backend_py"], {}, None),
        ("ranking and relevance for a recommendation product", ["search_rec"], {}, None),
        ("pytorch model training and deployment", ["ml"], {"remote": True}, None),
        ("data pipelines with airflow spark and a warehouse", ["data_eng"], {}, None),
        ("agents that call tools", ["llm"], {"job_type": "internship"}, None),
        ("kubernetes terraform infrastructure reliability", ["devops"], {}, None),
        ("react typescript user interface", ["frontend"], {}, None),
        ("dashboards sql reporting", ["analyst"], {}, None),
        ("java spring microservices kafka", ["backend_java"], {}, None),
        # ambiguous: personalisation should help
        ("software engineer internship", ["llm", "search_rec", "ml"], {"job_type": "internship"}, ai_profile),
        ("engineering roles", ["backend_py", "llm"], {"remote": True}, be_profile),
    ]
    with (ROOT / "evals" / "queries.jsonl").open("w") as f:
        for text, fams, flt, profile in queries:
            rec = {"query": text, "filters": flt, "relevant": rel(fams, **flt)}
            if profile:
                rec["profile"] = profile
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(jobs)} jobs and {len(queries)} eval queries")


if __name__ == "__main__":
    main()
