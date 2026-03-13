# GitHub Upload Guide

## 1. Create a new repository on GitHub

- Sign in to GitHub
- Click **New repository**
- Choose a repository name such as `reddit-insight-mining-and-clustering`
- Do **not** add a README, .gitignore, or license on GitHub since this project already includes them
- Click **Create repository**

## 2. Open Terminal in the project folder

```bash
cd /path/to/reddit-insight-mining-and-clustering
```

## 3. Initialize Git and make the first commit

```bash
git init
git add .
git commit -m "Initial commit"
```

## 4. Connect your local project to GitHub

Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your own values.

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

## 5. If Git asks you to authenticate

Use one of these methods:
- sign in through your browser if prompted
- or use a GitHub personal access token instead of your password

## 6. If you accidentally tracked large files before pushing

For example, if `reddit_posts.db` was already added:

```bash
git rm --cached reddit_posts.db
git add .gitignore
git commit -m "Remove local database from tracking"
git push -u origin main
```

## 7. Recommended files to keep in GitHub

Keep:
- all Python source files
- `README.md`
- `requirements.txt`
- `requirements_transformer_optional.txt`
- `config.json`
- supporting PDFs if required for your class

Usually do not keep:
- `.venv/`
- `reddit_posts.db`
- generated PNG files
- `lab8_outputs/`

## 8. Update after future changes

Whenever you edit the project later, use:

```bash
git add .
git commit -m "Describe your update"
git push
```
