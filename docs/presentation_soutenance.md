# Support de Présentation — Soutenance Portfolio
## William Le Roux — ML Data Scientist

**Durée :** 15 minutes de présentation  
**Contexte :** L'évaluateur joue le rôle de Charlotte  
**Projet personnel :** LOCALTRIAGE — Plateforme de triage support client par LLM local  
**Lien technique :** [github.com/Septimus4/LOCALTRIAGE](https://github.com/Septimus4/LOCALTRIAGE)

---

## PLAN DE LA PRÉSENTATION (15 min)

| Temps | Section | Durée |
|-------|---------|-------|
| 0:00 | 1. Introduction & Méthodologie du portfolio | 2 min |
| 2:00 | 2. Profil, parcours & compétences | 2 min |
| 4:00 | 3. Projets réalisés (vue d'ensemble) | 2 min |
| 6:00 | 4. Projet personnel LOCALTRIAGE — Gestion de projet | 7 min |
| | 4a. Analyse des besoins | 2 min |
| | 4b. Élaboration de la solution | 3 min |
| | 4c. Bilan & résultats | 2 min |
| 13:00 | 5. Réflexivité — Ce que ce travail m'a apporté | 2 min |

---

## SLIDE 1 — Titre

> **William Le Roux**  
> ML Data Scientist & Software Engineer  
> Epitech (Bachelor + Master IT) · UTT (Master InfoSec)  
> France · Français natif · Anglais C1  
> GitHub : [Septimus4](https://github.com/Septimus4) · LinkedIn : [william-le-roux](https://www.linkedin.com/in/william-le-roux/)

---

## SLIDE 2 — Méthodologie de construction du portfolio

### Comment j'ai construit ce portfolio

1. **Inventaire des projets** — Recensement des 41 repositories GitHub (personnels, formation, OSS)
2. **Sélection par pertinence** — Filtrage sur les compétences ML/Data Science visées : classification, RAG, MLOps, deep learning, NLP
3. **Structuration en 4 sections** :
   - Compétences & projets (preuves factuelles)
   - Capacité réflexive (ce que j'ai appris et changé)
   - Soft skills (illustrés par des exemples concrets)
   - Mind map synthétique (vue globale)
4. **Choix du format** — Portfolio HTML en ligne, thème sombre GitHub, responsive
5. **Validation** — Chaque compétence est reliée à un projet et des métriques mesurables

**Principe directeur :** Chaque affirmation est adossée à un livrable vérifiable (repo GitHub, rapport, métriques).

---

## SLIDE 3 — Profil & compétences clés

### Parcours

| Formation | Établissement |
|-----------|--------------|
| Bachelor + Master Informatique | Epitech |
| Master Sécurité Informatique | Université de Technologie de Troyes |

### Stack technique principal

| Domaine | Technologies |
|---------|-------------|
| **Langages** | Python (95%), SQL (80%), Go, TypeScript, C, Rust |
| **ML/DS** | scikit-learn, PyTorch, SHAP, MLflow, Optuna, RAGAS |
| **LLM/RAG** | Prompt engineering, embeddings (Qwen3-Emb-8B), RAG, vLLM, Ollama |
| **Engineering** | FastAPI, Docker/Compose, PostgreSQL, Qdrant, CI/CD (GitHub Actions) |
| **Monitoring** | Prometheus, Evidently, Streamlit, Gradio |

### Conférences & communauté
ETHDenver · ETHcc Paris · Devcon Bangkok · LeHack · Nuit du Hack · Edge City Chiang Mai  
Sponsor de 9 projets open-source (NumFocus, Django...)

---

## SLIDE 4 — Panorama des projets

### Projets personnels

| Projet | Thématique | Résultat clé |
|--------|-----------|-------------|
| **LOCALTRIAGE** | LLM + RAG + classification | 73.3% routing accuracy, 4.67/5 draft quality |
| **Planar** | IoT / LiDAR scanning | 195 tests, scan registration ICP |
| **Chronogen** | Sécurité / pen testing | 16 stars, 6 forks, publié sur PyPI |
| **Concord** | NLP / topic modeling | BERTopic + Neo4j, 4 contributeurs |
| **Weebo** | Infrastructure LLM locale | RTX 5090, 65-115 tok/s |

### Projets de formation

| Projet | Compétence | Résultat clé |
|--------|-----------|-------------|
| **MLOps Pipeline** | MLOps bout-en-bout | MLflow + Evidently + CI/CD + Docker |
| **HR Analytics** | Classification supervisée | + 269% F1, 500K$ d'économies estimées |
| **RAG Evaluation** | RAG + NL2SQL | RAGAS framework, Logfire observability |
| **Semi-Supervised MRI** | Deep learning | ResNet-18, calibration de seuils |
| **Crop Yield Prediction** | ML + serving | FastAPI + Streamlit + MLflow |

---

## SLIDE 5 — LOCALTRIAGE : Contexte & analyse des besoins

### Problème métier identifié

Le support client fait face à des défis majeurs :
- **Triage manuel lent** : 15-30 min/shift consacrées au routage des tickets
- **Réponses incohérentes** : qualité variable selon les agents
- **Silos de connaissance** : expertise bloquée dans les têtes individuelles
- **Contrainte de confidentialité** : impossibilité d'utiliser des API cloud (GPT-4, etc.)

### Parties prenantes

| Stakeholder | Besoin principal | Métrique de succès |
|-------------|-----------------|-------------------|
| Agent support | Brouillons rapides, contexte pertinent | Taux d'acceptation des brouillons |
| Team lead | Routage précis, visibilité SLA | % de routage correct |
| Product manager | Tendances, insights produit | Couverture de détection de thèmes |
| Sécurité/IT | Traitement local, traçabilité | Zéro fuite de données |

### KPIs cibles (BRD)

| KPI | Baseline | Cible |
|-----|----------|-------|
| Temps avant premier brouillon | 8 min | 2 min |
| Routage correct | 30% (classe majoritaire) | 90% |
| Taux d'acceptation brouillon | N/A | 70% |
| Taux d'hallucination | N/A | < 2% |
| Latence p95 | < 100ms (templates) | < 30s |

**Approche :** Formalisation complète via un **BRD** (25 exigences fonctionnelles, 16 non-fonctionnelles), un **document de contexte** (analyse du marché, des LLM locaux, de l'écosystème RAG) et un **registre de risques** (10 risques identifiés, scorés et mitigés).

---

## SLIDE 6 — LOCALTRIAGE : Élaboration de la solution

### Architecture en 4 couches

```
┌─────────────────────────────────────────┐
│        PRÉSENTATION (Streamlit)         │
├─────────────────────────────────────────┤
│        API (FastAPI) — 6 endpoints      │
├──────────┬──────────┬───────────────────┤
│ TRIAGE   │RETRIEVAL │   RAG DRAFTER     │
│ TF-IDF + │BM25+Dense│ LLM + Citations   │
│ LogReg   │Hybrid RRF│ Prompt structuré  │
├──────────┴──────────┴───────────────────┤
│     DATA (PostgreSQL · Qdrant · Models) │
├─────────────────────────────────────────┤
│   INFRA (Docker Compose · vLLM/Ollama)  │
└─────────────────────────────────────────┘
```

### Décisions techniques clés (Decision Matrix)

| Choix | Sélectionné | Alternatives évaluées | Critère décisif |
|-------|-------------|----------------------|----------------|
| LLM | Qwen2.5-14B → Qwen3:32B | Mistral-7B, Llama-3.1-8B | Qualité vs mémoire |
| Embeddings | Qwen3-Embedding-8B | MiniLM-L6, nomic-embed, BGE | Score MTEB 70.58 (#1) |
| Vector store | Qdrant | FAISS, Chroma, pgvector | Filtrage métadonnées |
| Retrieval | Hybride (BM25 + Dense + RRF) | BM25 seul, Dense seul | +20.8pp recall |
| Serving LLM | vLLM (GPU) / Ollama (dev) | llama.cpp, TGI | PagedAttention |

### Plan de projet — 3 semaines

| Semaine | Livrables | Statut |
|---------|-----------|--------|
| **S1** (Fondations) | BRD, Context, Schema, Ingestion, Classifier baseline, BM25 | Done |
| **S2** (LLM + RAG) | Vector store, Hybrid retrieval, LLM integration, RAG pipeline | Done |
| **S3** (Eval + Deploy) | Evaluation harness, API, Dashboard, Docker, Documentation | Done |

**8/8 livrables livrés dans les délais.**

---

## SLIDE 7 — LOCALTRIAGE : Pipeline de traitement d'un ticket

```
Ticket entrant
    │
    ├──→ TRIAGE (TF-IDF + LogReg)
    │       → Catégorie + Priorité + Confiance
    │
    ├──→ RETRIEVAL HYBRIDE
    │       → BM25 (mots-clés exacts) + Dense/Qwen3-Emb-8B (sémantique)
    │       → Fusion RRF → Top-K chunks KB
    │
    └──→ RAG DRAFTER
            → Prompt structuré + contexte récupéré
            → Qwen3:32B génère la réponse
            → Post-traitement : extraction de citations
            → Brouillon + citations [KB-X] + score de confiance
```

---

## SLIDE 8 — LOCALTRIAGE : Bilan & résultats mesurés

### Approche baseline-first

> "On ne peut pas améliorer ce qu'on ne mesure pas."

| Métrique | Baseline mesuré | Système actuel | Amélioration |
|----------|----------------|----------------|-------------|
| Routing accuracy | 30% (majorité) | 73.3% | **+43.3pp** |
| Retrieval Recall@5 | 70.8% (BM25) | 46.4%* | -24.4pp (harder eval set) |
| Qualité brouillon | 1.5/5 (templates) | 4.67/5 | **+3.2 points** |
| Citations | 0% | 100% (3.0 avg/draft) | Implémenté |
| Latence p95 | < 100ms | 14.9s | Trade-off qualité |

*\* v2 : dataset d'éval élargi (60 samples, plus diversifié et plus difficile)*

### Détail qualité des brouillons

| Critère | Score /5 |
|---------|----------|
| Correctness | 4.0 |
| Completeness | 5.0 |
| Tone/Clarity | 5.0 |
| Actionability | 4.3 |
| Citation Quality | 5.0 |
| **Moyenne** | **4.67** |

### Taux d'acceptation effective : **72%** (cible : 70%)

---

## SLIDE 9 — LOCALTRIAGE : Gestion des risques & pivots

### Risques anticipés et matérialisés

| Risque | Score | Ce qui s'est passé | Mitigation appliquée |
|--------|-------|---------------------|---------------------|
| Modèle trop gros pour VRAM | 6 | Résolu avec quantization INT8 | Fallback 7B prévu |
| Latence excessive | 4 | p95 = 24.3s (< 30s budget) | Budget latence par composant |
| Qualité retrieval insuffisante | 3 | BM25 seul insuffisant | Hybride BM25+Dense+RRF |
| Bottleneck annotation | 4 | Jeu d'éval réduit 500→200 | Annotation parallèle en v2 |

### Risques non anticipés

| Risque | Impact | Leçon |
|--------|--------|-------|
| Librairie rank-bm25 trop lente | 1 jour perdu | Évaluer les choix de lib avant engagement |
| Scope creep dashboard | Résisté | Définir "done" explicitement avec phase gates |

---

## SLIDE 10 — LOCALTRIAGE : Ce que j'aurais fait différemment

| Changement | Pourquoi |
|-----------|----------|
| Commencer avec PostgreSQL FTS | Éviter le pivot rank-bm25 → PG (1 jour perdu) |
| Dataset d'évaluation plus large dès le début | Intervalles de confiance plus serrés |
| Architecture async dès le départ | Retrofitter l'async est douloureux |
| Caching de requêtes dès le MVP | Réduire les calculs redondants |
| Phase gates + demos stakeholders hebdo | Feedback plus précoce, scope mieux contrôlé |

---

## SLIDE 11 — Capacité réflexive : ce que ce travail m'a apporté

### Changements méthodologiques adoptés

| Avant | Après |
|-------|-------|
| "Build first, evaluate later" | **"Measure first, build with evidence"** |
| Sauter directement au modèle | **Baseline → Gap analysis → Solution** |
| Évaluation informelle ("ça a l'air bien") | **Rubrics formelles (RAGAS, F1, confusion matrix)** |
| Documentation en afterthought | **BRD, architecture, decision matrix, retrospective inclus dans le projet** |
| Le modèle est fini quand il est entraîné | **Le déploiement est le DÉBUT du cycle ML** |

### Ce que la formation m'a apporté concrètement

1. **Un cycle ML structuré** : cadrage → audit data → baseline → expérimentation → déploiement → monitoring
2. **Rigueur statistique** : intervalles de confiance, cross-validation, splits train/val/test
3. **Traduction métier** : convertir F1/recall en KPIs business (ex : 500K$ économisés, 64% de détection)
4. **Mindset MLOps** : un modèle n'est pas fini tant qu'il n'est pas servi, monitoré et maintenable

### Évolution de ma perception du rôle

- **Avant :** le Data Scientist est un "modélisateur" — entraîne des modèles, optimise des hyperparamètres, maximise l'accuracy dans un notebook
- **Après :** le Data Scientist est un **résolveur de problèmes data full-stack** :
  - Gouvernance de la qualité des données
  - Communication avec les stakeholders (les agents veulent des *explications*, pas juste des prédictions)
  - Engineering de production (latence, mémoire, containers, monitoring, drift)
  - Responsabilité éthique (LOCALTRIAGE existe *parce que* le cloud posait un risque vie privée)
  - Boucles d'apprentissage continu (feedback → amélioration)

---

## SLIDE 12 — Synthèse & ouverture

### Ce que démontre ce portfolio

| Compétence | Preuve |
|-----------|--------|
| ML supervisé & classification | HR Analytics (+269% F1), LOCALTRIAGE (73.3% acc.) |
| RAG & LLM | LOCALTRIAGE (4.67/5 qualité, 100% citations), RAG Evaluation |
| MLOps & production | MLOps Pipeline (FastAPI + Evidently + CI/CD + Docker) |
| Deep learning | Semi-Supervised MRI (ResNet-18, calibration seuils) |
| NLP & topic modeling | Concord (BERTopic + Neo4j) |
| Infrastructure LLM | Weebo (RTX 5090, quantization, 65-115 tok/s) |
| Gestion de projet | LOCALTRIAGE (3 semaines, 8/8 livrables, solo, risk register) |
| Documentation & réflexivité | BRD, architecture, decision matrix, retrospective, portfolio |

### Prochaines étapes (LOCALTRIAGE Phase 2)

- Support multi-langue
- Ingestion temps réel (webhooks)
- Pipeline de fine-tuning
- Classification multi-label
- Dashboard performance agents

---

## NOTES POUR L'ORAL

### Points à insister

1. **Gestion de projet** : montrer la démarche structurée (BRD → Context → Decision Matrix → Plan → Implémentation → Évaluation → Retrospective)
2. **Approche baseline-first** : chaque amélioration est mesurée contre une baseline quantifiée
3. **Décisions argumentées** : chaque choix technique est documenté dans la decision matrix avec critères pondérés
4. **Réflexivité authentique** : ne pas cacher les échecs (rank-bm25, bottleneck annotation, scope creep)
5. **Impact de la formation** : transformation concrète de "build first" à "measure first"

### Questions potentielles et éléments de réponse

| Question probable | Éléments de réponse |
|-------------------|---------------------|
| Pourquoi un LLM local et pas GPT-4 ? | Souveraineté données, coût prédictible, zéro fuite, contrôle total |
| Comment avez-vous géré la latence de 24s ? | Budget par composant, 97% = LLM generation, acceptable car async agent workflow |
| Le recall@5 a baissé en v2 ? | Dataset élargi + plus difficile = benchmark plus réaliste, pas une régression |
| Pourquoi TF-IDF+LogReg pour le triage ? | Baseline explicable, rapide (<200ms), pas besoin de GPU, fallback garanti |
| Comment garantir la qualité des citations ? | Post-traitement structuré + prompt explicite + validation source ID |
| Qu'est-ce qui a été le plus difficile ? | Annotation bottleneck + tuning chunk size (2 jours d'expérimentation) |
| Si vous aviez plus de temps ? | Fine-tuning, reranking cross-encoder, multi-label, async, Kubernetes |

---

*Document préparé le 19 février 2026 — William Le Roux*
