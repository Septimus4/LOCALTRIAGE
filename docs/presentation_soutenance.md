# Support de Presentation -- Soutenance Portfolio
## William Le Roux -- ML Data Scientist

**Duree :** 15 minutes de presentation  
**Contexte :** L'evaluateur joue le role de Charlotte  
**Projet personnel :** LOCALTRIAGE -- Plateforme de triage support client par LLM local  
**Lien technique :** [github.com/Septimus4/LOCALTRIAGE](https://github.com/Septimus4/LOCALTRIAGE)

---

## PLAN DE LA PRESENTATION (15 min)

| Temps  | Partie | Slides | Duree |
|--------|--------|--------|-------|
| 0:00   | **Partie 1 -- Contexte & Pilotage** | Slides 2-5 | 6 min |
|        | Contexte organisationnel & probleme metier | Slide 2 | 1.5 min |
|        | Collecte des besoins & BRD | Slide 3 | 1.5 min |
|        | Appui strategique & decision matrix | Slide 4 | 1.5 min |
|        | Pilotage du projet (delais, couts, livrables) | Slide 5 | 1.5 min |
| 6:00   | **Partie 2 -- Realisation & Resultats** | Slides 6-9 | 7 min |
|        | Architecture & stack technique | Slide 6 | 2 min |
|        | CI, tests & qualite | Slide 7 | 2 min |
|        | Resultats mesures | Slide 8 | 2 min |
|        | Retrospective & lecons | Slide 9 | 1 min |
| 13:00  | **Partie 3 -- Portfolio** | Slides 10-11 | 2 min |
|        | Le portfolio : construction & contenu | Slide 10 | 1 min |
|        | Ouverture du portfolio & merci | Slide 11 | 1 min |

---

# ============================================================
# PARTIE 1 -- CONTEXTE & GESTION DE PROJET
# ============================================================

---

## SLIDE 1 -- Couverture

> **William Le Roux**
> ML Data Scientist & Software Engineer
> Epitech (Bachelor + Master IT) -- UTT (Master InfoSec)
> France -- Francais natif -- Anglais C1
> GitHub : Septimus4 -- LinkedIn : william-le-roux

Plan en 3 parties :
- **Partie 1 :** Contexte & Pilotage
- **Partie 2 :** Realisation & Resultats
- **Partie 3 :** Portfolio

---

## SLIDE 2 -- Contexte organisationnel & probleme metier

### Contexte du secteur

| Chiffre | Description |
|---------|------------|
| +50-200% | Croissance annuelle du volume de tickets |
| 40% | Turnover annuel des agents support |
| 68% | Entreprises : vie privee = frein n.1 pour l'IA |
| <1h | Attente client pour 1ere reponse |

### Problemes identifies

- **Triage manuel lent :** 15-30 min/shift sur le routage
- **Reponses incoherentes :** qualite variable selon les agents
- **Silos de connaissance :** expertise bloquee dans les tetes
- **Vie privee :** impossible d'utiliser des API cloud (GPT-4, etc.)

### Analyse des parties prenantes

| Partie prenante | Besoin principal | KPI de succes |
|----------------|-----------------|---------------|
| Agent support | Brouillons rapides, contexte pertinent | Taux d'acceptation |
| Team lead | Routage precis, visibilite SLA | % routage correct |
| Product manager | Tendances, insights produit | Detection de themes |
| Securite / IT | Traitement local, tracabilite | Zero fuite de donnees |

### Opportunite business

> Deployer un systeme LLM local pour reduire le temps de reponse de 40-60%, ameliorer le routage a >90%, et garantir la souverainete des donnees.

Tags : Self-hosted, LLM local, Zero data egress, RAG + citations

---

## SLIDE 3 -- Collecte des besoins metiers & formalisation (BRD)

### KPIs cibles definis dans le BRD

| KPI | Baseline | Cible |
|-----|----------|-------|
| Temps avant 1er brouillon | 8 min | 2 min |
| Routage correct | 30% (classe majoritaire) | 90% |
| Taux d'acceptation brouillon | N/A | 70% |
| Taux d'hallucination | N/A | < 2% |
| Latence p95 (E2E) | < 100ms (templates) | < 30s |
| Detection themes emergents | 5 jours | 1 jour |
| Disponibilite | -- | 99.5% |

### Exigences formalisees

- **25** exigences fonctionnelles
- **16** exigences non-fonctionnelles
- **10** risques identifies & scores

### Livrables de cadrage produits

1. **BRD** (Business Requirements Document) -- 25 FR, 16 NFR, KPIs quantifies, stakeholder analysis, acceptance criteria
2. **Context Analysis** -- Paysage concurrentiel, ecosysteme LLM locaux (2026), patterns RAG, barrieres d'adoption, quantification
3. **Risk Register** -- 10 risques (probabilite x impact), mitigations planifiees, contingency plans, owners assignes
4. **Decision Matrix** -- Criteres ponderes (qualite 30%, vitesse 25%, memoire 20%, licence 15%, ecosysteme 10%), score composite

> Chaque decision est tracable : du besoin metier au KPI, du KPI a l'exigence, de l'exigence au choix technique.

---

## SLIDE 4 -- Appui strategique & methodologique pour la prise de decision

### Decision Matrix -- choix argumentes

| Composant | Selection | Alternatives evaluees | Critere decisif |
|-----------|-----------|----------------------|----------------|
| LLM | Qwen3:32B | Mistral-7B, Llama-3.1-8B, Qwen2.5-14B | Qualite vs memoire |
| Embeddings | Qwen3-Emb-8B | MiniLM-L6, nomic-embed, BGE | MTEB #1 (70.58) |
| Vector DB | Qdrant | FAISS, Chroma, pgvector | Filtrage metadonnees |
| Retrieval | Hybride + RRF | BM25 seul, Dense seul | +20.8pp recall |
| LLM Serving | vLLM / Ollama | llama.cpp, TGI | PagedAttention |
| API | FastAPI | Flask, Django REST | Async + OpenAPI |

### Methode de scoring

Chaque composant evalue sur 5 criteres ponderes :
- Qualite : 30%
- Vitesse : 25%
- Memoire : 20%
- Licence : 15%
- Ecosysteme : 10%

Score composite = somme ponderee des notes /100 par critere.

### Gestion proactive des risques

| Risque | Score | Mitigation appliquee |
|--------|-------|---------------------|
| Modele trop gros / VRAM | 6 | Quantization Q4_K_M + fallback 7B |
| Latence excessive | 4 | Budget latence par composant |
| Retrieval insuffisant | 3 | Hybride BM25 + Dense + RRF |
| Qualite donnees KB | 4 | Pipeline de validation |

### Tracabilite complete

Besoin metier --> Exigence BRD --> Choix technique (Decision Matrix) --> KPI mesure (Evaluation Harness)

---

## SLIDE 5 -- Pilotage du projet -- delais, couts, livrables, performance

### Plan en 3 semaines -- 3 milestones

**S1 -- Fondations**
- BRD, Context, architecture
- Schema BDD + ingestion
- Classifier baseline (TF-IDF)
- Retrieval BM25
- [Milestone 1]

**S2 -- LLM & RAG**
- Vector store + embeddings
- Retrieval hybride + RRF
- Integration LLM Qwen3:32B
- Pipeline RAG + citations
- [Milestone 2]

**S3 -- Eval & Deploy**
- Evaluation harness
- API FastAPI (6 endpoints)
- Dashboard Streamlit
- Docker + documentation
- [Milestone 3]

### Bilan

| Indicateur | Valeur |
|-----------|--------|
| Livrables livres | 8/8 |
| Delai | 3 semaines -- dans les delais |
| Developpement | Solo |
| Cout cloud | 0 EUR (100% local) |

### Livrables produits

- [OK] Pipeline d'ingestion
- [OK] Audit baseline & mesures
- [OK] Routage automatise
- [OK] Drafting RAG + citations
- [OK] Dashboard analytics
- [OK] Evaluation harness
- [OK] Deploiement Docker
- [OK] Documentation complete

### Chemin critique

Schema --> Ingestion --> Vector Store --> Hybrid Retrieval --> RAG --> API

Parallelisation : baselines developpees en parallele de l'integration LLM.

---

# ============================================================
# PARTIE 2 -- REALISATION TECHNIQUE & RESULTATS
# ============================================================

---

## SLIDE 6 -- Architecture de la solution & pipeline

### Architecture en couches

```
+------------------------------------------+
|      PRESENTATION -- Streamlit Dashboard  |
+------------------------------------------+
|      API -- FastAPI (6 endpoints)         |
+-------------+------------+---------------+
| TRIAGE      | RETRIEVAL   | RAG DRAFTER   |
| TF-IDF +    | BM25 +      | LLM + Prompt  |
| LogReg      | Dense + RRF | + Citations   |
+-------------+------------+---------------+
|      DATA -- PostgreSQL / Qdrant / Models |
+------------------------------------------+
|      INFRA -- Docker Compose / vLLM       |
+------------------------------------------+
```

### Pipeline de traitement d'un ticket

Ticket entrant --> Triage (cat + prio) --> Retrieval (BM25+Dense) --> RRF (fusion) --> LLM (Qwen3:32B) --> Brouillon + citations

### Stack technique

| Couche | Technologie |
|--------|-------------|
| LLM | Qwen3:32B (Q4_K_M) |
| Embeddings | Qwen3-Emb-8B (4096d) |
| Vector DB | Qdrant |
| BDD | PostgreSQL |
| Retrieval | Hybride BM25+Dense+RRF |
| Serving | Ollama / vLLM |
| API | FastAPI (6 endpoints) |
| UI | Streamlit |
| Infra | Docker Compose |
| GPU | RTX 5090 (32 GB VRAM) |

---

## SLIDE 7 -- Strategie de test, CI/CD & assurance qualite

### Pyramide de tests

| Niveau | Tests | Couverture |
|--------|-------|-----------|
| Tests unitaires | 65 passes | Triage, retrieval, drafter, ingestion, data validation |
| Tests end-to-end (E2E) | 26 passes | API endpoints, triage flow, draft flow, health checks |
| Evaluation harness (ML) | 60 samples eval | Routing accuracy, recall@5, draft quality rubric (5 criteres), latence |
| Tests Postman (collection) | 3 envs | Collection exportable, env local + staging, scripts pre/post-request |

### Bilan couverture

- Unit tests : 65
- E2E tests : 26
- Total tests : 91
- Echecs : **0**

### Outils & pratiques

| Domaine | Outil / Pratique |
|---------|-----------------|
| Framework test | pytest + fixtures |
| E2E / API | httpx + Postman collection |
| Eval ML | Harness custom (rubric LLM-as-judge) |
| Containerisation | Docker Compose (4 services) |
| Linting | Ruff + type hints |
| Versioning | Git + GitHub |

### Monitoring

Prometheus, Streamlit dashboard, Health endpoint

---

## SLIDE 8 -- Resultats mesures -- approche baseline-first

> "On ne peut pas ameliorer ce qu'on ne mesure pas." -- Chaque metrique est comparee a une baseline quantifiee.

### KPIs principaux

| KPI | Valeur | Delta |
|-----|--------|-------|
| Routing accuracy | 73.3% | +43.3pp vs baseline |
| Qualite brouillons | 3.63/5 | +2.1 vs templates |
| Taux d'acceptation | 72% | cible 70% -- atteint |
| Latence p95 | 10.8s | < 30s budget -- atteint |

### Comparaison baseline vs systeme

| Metrique | Baseline | Actuel | Delta |
|----------|----------|--------|-------|
| Routing accuracy | 30% (majorite) | 73.3% | +43.3pp |
| Recall@5 | 70.8% (BM25) | 46.4% | harder eval set |
| Qualite drafts | 1.5/5 (template) | 3.63/5 | +2.1 |
| Citations | 0% | 100% | 2.8 avg/draft |
| Hallucinations | N/A | 0% | Pass |
| Detection P1 | 42% recall | 89% | +47pp |

### Detail qualite des brouillons (rubric)

| Critere | Score /5 |
|---------|----------|
| Correctness | 3.8 |
| Completeness | 3.2 |
| Tone / Clarity | 4.0 |
| Actionability | 3.2 |
| Citation Quality | 4.0 |
| **Moyenne** | **3.63** |

Evaluation par LLM-as-judge (Qwen3:32B, temperature=0, critique-first rubric) sur 5 tickets representatifs couvrant 5 categories.

---

## SLIDE 9 -- Retrospective -- pivots, lecons et reflexivite

### Risques materialises & pivots

1. **rank-bm25 trop lent en prod** (1 jour perdu) -- Pivot vers PostgreSQL FTS. Lecon : evaluer les libs contre les contraintes prod avant engagement.
2. **Bottleneck d'annotation** (eval set reduit) -- 500 --> 200 samples. Lecon : demarrer l'annotation en parallele des le jour 1.
3. **Scope creep dashboard** (resiste) -- Lecon : phase gates strictes + definition de "done" explicite.

### Ce que j'aurais fait differemment

- Commencer avec PostgreSQL FTS directement
- Dataset d'evaluation plus large des le depart
- Architecture async des le jour 1
- Caching de requetes dans le MVP
- Demos stakeholders hebdomadaires

### Evolution methodologique : Avant vs Apres

| Avant | Apres |
|-------|-------|
| "Build first, evaluate later" | **"Measure first, build with evidence"** |
| Sauter au modele directement | **Baseline --> Gap analysis --> Solution** |
| Evaluation informelle | **Rubrics formelles (RAGAS, F1)** |
| Documentation en afterthought | **BRD, archi, decision matrix, retro** |
| Fini = entraine | **Deploiement = DEBUT du cycle** |

> Le Data Scientist n'est pas un modelisateur mais un **resolveur de problemes data full-stack** : gouvernance, communication stakeholders, engineering de production, ethique, apprentissage continu.

---

# ============================================================
# PARTIE 3 -- PORTFOLIO
# ============================================================

---

## SLIDE 10 -- Le portfolio -- construction & contenu

### Demarche de construction

1. **Inventaire** -- Recensement de 41 repositories GitHub (personnels, formation, OSS)
2. **Selection par pertinence** -- Filtrage sur competences ML/DS : classification, RAG, MLOps, deep learning, NLP
3. **Structuration en 4 sections** -- Competences & projets, capacite reflexive, soft skills, mind map
4. **Validation croisee** -- Chaque competence est reliee a un projet et a des metriques mesurables

### Structure du portfolio

| Section | Contenu |
|---------|--------|
| Competences & Projets | 12 projets detailles avec stack, metriques, liens GitHub |
| Capacite reflexive | Erreurs, lecons, evolution du regard sur le metier |
| Soft skills | 8 competences illustrees par des exemples concrets |
| Mind map | Vue synthetique de l'ensemble du profil et des connexions |

### Competences demontrees

| Competence | Preuve |
|-----------|--------|
| ML supervise | HR Analytics (+269% F1), LOCALTRIAGE (73.3%) |
| RAG & LLM | LOCALTRIAGE (3.63/5, 100% citations) |
| MLOps | Pipeline (FastAPI + Evidently + CI/CD) |
| Deep learning | Semi-Supervised MRI (ResNet-18) |
| Gestion de projet | LOCALTRIAGE (3 sem, 8/8 livrables, solo) |

> Principe directeur : chaque affirmation est adossee a un livrable verifiable.

---

## SLIDE 11 -- Ouverture du Portfolio & Merci

> Le portfolio HTML presente l'ensemble des projets, competences et reflexions. Je vous propose de le parcourir ensemble.

**Liens :**
- Portfolio HTML en ligne
- github.com/Septimus4/LOCALTRIAGE

**Merci -- Questions & echanges**

---

# ============================================================
# NOTES POUR L'ORAL
# ============================================================

---

### Points a insister

1. **Gestion de projet** : montrer la demarche structuree (BRD --> Context --> Decision Matrix --> Plan --> Implementation --> Evaluation --> Retrospective)
2. **Approche baseline-first** : chaque amelioration est mesuree contre une baseline quantifiee
3. **Decisions argumentees** : chaque choix technique est documente dans la decision matrix avec criteres ponderes
4. **Reflexivite authentique** : ne pas cacher les echecs (rank-bm25, bottleneck annotation, scope creep)
5. **Impact de la formation** : transformation concrete de "build first" a "measure first"

### Questions potentielles et elements de reponse

| Question probable | Elements de reponse |
|-------------------|---------------------|
| Pourquoi un LLM local et pas GPT-4 ? | Souverainete donnees, cout predictible, zero fuite, controle total |
| Comment avez-vous gere la latence de 10.8s ? | Budget par composant, 97% = LLM generation, acceptable car async agent workflow |
| Le recall@5 a baisse en v2 ? | Dataset elargi + plus difficile = benchmark plus realiste, pas une regression |
| Pourquoi TF-IDF+LogReg pour le triage ? | Baseline explicable, rapide (<200ms), pas besoin de GPU, fallback garanti |
| Comment garantir la qualite des citations ? | Post-traitement structure + prompt explicite + validation source ID |
| Qu'est-ce qui a ete le plus difficile ? | Annotation bottleneck + tuning chunk size (2 jours d'experimentation) |
| Si vous aviez plus de temps ? | Fine-tuning, reranking cross-encoder, multi-label, async, Kubernetes |

---

*Document prepare le 19 fevrier 2026 -- William Le Roux*
