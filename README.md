# Reddit Insight Mining and Clustering

This project builds a Reddit data pipeline for scraping, preprocessing, vectorizing, clustering, and visualizing posts. It also includes a Lab 8 extension for comparing multiple document embedding methods.

## Project Overview

The repository supports two main workflows:

1. **Original final project pipeline**
   - Scrape Reddit posts into SQLite
   - Clean and enrich post content
   - Generate document embeddings
   - Cluster posts and inspect clusters visually

2. **Lab 8 embedding comparison extension**
   - Train and compare three Doc2Vec configurations
   - Build Word2Vec-based semantic-bin document vectors
   - Cluster all document vectors with KMeans
   - Evaluate methods with silhouette score and average intra-cluster similarity

## Repository Structure

- `reddit_scraper.py` - scrape Reddit posts into SQLite
- `sqlite_db.py` - database schema creation and inserts
- `process.py` - raw text cleaning helpers
- `post_process.py` - post-cleaning, OCR options, keyword/topic enrichment
- `vector.py` - original embedding pipeline
- `cluster.py` - original clustering pipeline
- `automate.py` - end-to-end automation helper
- `summary.py` - quick database summary utility
- `view.py` - SQL inspection utility
- `lab8_pipeline.py` - Lab 8 embedding comparison pipeline
- `make_lab8_pdfs.py` - helper for Lab 8 PDF generation
- `config.json` - project configuration
- `requirements.txt` - core dependencies
- `requirements_transformer_optional.txt` - optional transformer dependency
- `README.pdf` - submission-friendly run guide
- `meeting_notes_L8_team_name.pdf` - meeting-notes template
- `lab8_assignment.pdf` - lab assignment reference

## Python Version

This project was adjusted to remain compatible with **Python 3.6**.

Examples of compatibility-related changes include:
- replacing newer built-in generic annotations such as `list[str]` with Python-3.6-safe typing forms like `List[str]`
- avoiding syntax that fails in older lab environments

## Installation

Create and activate a virtual environment, then install the core dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional transformer-based embedding support:

```bash
pip install -r requirements_transformer_optional.txt
```

## How to Run the Main Pipeline

### 1. Scrape Reddit posts

```bash
python reddit_scraper.py --posts 5000
```

### 2. Post-process scraped data

```bash
python post_process.py --db reddit_posts.db --skip-ocr
```

### 3. Create embeddings

```bash
python vector.py --db reddit_posts.db --method transformer
```

### 4. Cluster documents

```bash
python cluster.py --db reddit_posts.db --clusters auto
```

## How to Run the Lab 8 Extension

```bash
python lab8_pipeline.py --db reddit_posts.db --out lab8_outputs --clusters 5
```

This generates outputs such as:
- per-method JSON results
- PCA cluster plots
- `lab8_summary.json`
- `lab8_report.md`

## Evaluation Summary

The Lab 8 extension compares six embedding configurations:
- Doc2Vec at 50, 100, and 200 dimensions
- Word2Vec plus semantic bins at 50, 100, and 200 dimensions

All document vectors are L2-normalized before clustering so that KMeans approximates cosine-based clustering. Methods are evaluated using:
- cosine silhouette score
- average intra-cluster similarity
- qualitative inspection of titles and cluster terms


## Visual Results

To make the repository easier to review on GitHub, the README includes sample visuals from the project workflow and clustering analysis.

### Cluster Scatter Plot

This figure shows the cluster distribution from the project visualization pipeline.

![Cluster Scatter Plot](assets/images/cluster_scatter.png)

### Cluster Search Result View

This figure shows an example cluster search / inspection output used to interpret grouping quality.

![Cluster Search Result](assets/images/cluster_search_result.png)

## Notes for GitHub

This GitHub-ready version excludes large generated data files and local artifacts that should usually not be committed, such as:
- SQLite databases
- generated plots
- local virtual environments
- cache files

If your instructor requires generated outputs in the repository, you can remove those patterns from `.gitignore` before pushing.

## Submission Notes

The included PDFs can still be used for class submission. Update the meeting-notes template with your final team information before submitting.
