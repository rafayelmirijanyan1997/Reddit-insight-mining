import argparse
import json
import logging
import math
import os
import re
import sqlite3
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

DEFAULT_DB = 'reddit_posts.db'
DEFAULT_OUTPUT_DIR = 'lab8_outputs'
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('lab8')

STOP_WORDS = {
    'the', 'and', 'for', 'that', 'with', 'this', 'are', 'was', 'were', 'been',
    'but', 'not', 'you', 'your', 'all', 'can', 'had', 'her', 'his', 'him',
    'how', 'its', 'may', 'new', 'now', 'old', 'see', 'way', 'who', 'did',
    'get', 'has', 'have', 'just', 'more', 'also', 'from', 'they', 'will',
    'what', 'when', 'where', 'which', 'their', 'there', 'these', 'those',
    'would', 'about', 'could', 'other', 'than', 'then', 'them', 'into',
    'some', 'such', 'only', 'over', 'very', 'after', 'before', 'being',
    'between', 'both', 'each', 'because', 'does', 'during', 'should', 'while',
    'here', 'most', 'much', 'many', 'well', 'back', 'like', 'make', 'made',
    'know', 'think', 'still', 'even', 'take', 'come', 'want', 'say', 'said',
    'use', 'used', 'first', 'going', 'people', 'thing', 'things', 'really',
    'good', 'great', 'right', 'look', 'long', 'little', 'big', 'keep', 'let',
    'put', 'give', 'tell', 'need', 'every', 'own', 'through', 'our', 'out',
    'any', 'time', 'day', 'too', 'don', 'one', 'two', 'off', 'got', 'why',
    'http', 'https', 'www', 'com', 'reddit', 'post', 'posts'
}


DOC2VEC_CONFIGS = [
    {'name': 'doc2vec_50d', 'vector_size': 50, 'epochs': 30, 'min_count': 2, 'window': 8, 'dm': 1},
    {'name': 'doc2vec_100d', 'vector_size': 100, 'epochs': 40, 'min_count': 2, 'window': 10, 'dm': 1},
    {'name': 'doc2vec_200d', 'vector_size': 200, 'epochs': 50, 'min_count': 2, 'window': 10, 'dm': 1},
]


WORD2VEC_CONFIGS = [
    {'name': 'w2v_bins_50d', 'vector_size': 50, 'epochs': 20, 'min_count': 2, 'window': 5, 'bins': 50},
    {'name': 'w2v_bins_100d', 'vector_size': 100, 'epochs': 25, 'min_count': 2, 'window': 5, 'bins': 100},
    {'name': 'w2v_bins_200d', 'vector_size': 200, 'epochs': 30, 'min_count': 2, 'window': 5, 'bins': 200},
]


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def tokenize(text):
    return re.findall(r"\b[a-z]{3,}\b", (text or '').lower())


def load_posts(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT post_id, subreddit, COALESCE(cleaned_title, title, '') AS title_text, "
        "COALESCE(cleaned_selftext, selftext, '') AS body_text, COALESCE(keywords, '') AS keywords "
        "FROM posts"
    ).fetchall()
    conn.close()

    posts = []
    for row in rows:
        combined = (row['title_text'] + ' ' + row['body_text']).strip()
        if not combined:
            continue
        posts.append({
            'post_id': row['post_id'],
            'subreddit': row['subreddit'],
            'title': row['title_text'].strip() or '(no title)',
            'body': row['body_text'].strip(),
            'text': combined,
            'tokens': tokenize(combined),
            'keywords': [k.strip() for k in (row['keywords'] or '').split(',') if k.strip()],
        })
    if not posts:
        raise RuntimeError('No usable posts found in database.')
    return posts


def normalize_vectors(vectors):
    arr = np.asarray(vectors, dtype=np.float32)
    return normalize(arr)


def cosine_cluster(vectors, k):
    norm_vectors = normalize_vectors(vectors)
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(norm_vectors)
    sil = silhouette_score(norm_vectors, labels, metric='cosine') if len(set(labels)) > 1 else -1.0
    return labels, norm_vectors, sil


def top_words_from_texts(texts, top_n=10):
    counts = Counter()
    for text in texts:
        counts.update([t for t in tokenize(text) if t not in STOP_WORDS])
    return [w for w, _ in counts.most_common(top_n)]


def average_intra_cluster_similarity(vectors, labels):
    norm = normalize_vectors(vectors)
    sims = []
    for cluster_id in sorted(set(labels)):
        idx = np.where(labels == cluster_id)[0]
        if len(idx) < 2:
            continue
        sub = norm[idx]
        centroid = np.mean(sub, axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
        sims.extend(np.dot(sub, centroid_norm).tolist())
    return float(np.mean(sims)) if sims else 0.0


def cluster_descriptions(posts, labels, sample_size=4):
    desc = []
    for cluster_id in sorted(set(labels)):
        idx = np.where(labels == cluster_id)[0]
        texts = [posts[i]['text'] for i in idx]
        titles = [posts[i]['title'] for i in idx][:sample_size]
        desc.append({
            'cluster_id': int(cluster_id),
            'size': int(len(idx)),
            'top_words': top_words_from_texts(texts, top_n=8),
            'sample_titles': titles,
        })
    return desc


def plot_pca(vectors, labels, title, output_path):
    norm_vectors = normalize_vectors(vectors)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(norm_vectors)
    k = len(set(labels))
    cmap = plt.cm.get_cmap('tab10', k)
    plt.figure(figsize=(10, 7))
    for cluster_id in range(k):
        mask = labels == cluster_id
        plt.scatter(coords[mask, 0], coords[mask, 1], s=14, alpha=0.65, c=[cmap(cluster_id)], label='C%d' % cluster_id)
    plt.title(title)
    plt.xlabel('PC1 (%.1f%% variance)' % (pca.explained_variance_ratio_[0] * 100.0))
    plt.ylabel('PC2 (%.1f%% variance)' % (pca.explained_variance_ratio_[1] * 100.0))
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def train_doc2vec(posts, config):
    tagged = [TaggedDocument(words=(p['tokens'] or ['empty']), tags=[str(i)]) for i, p in enumerate(posts)]
    model = Doc2Vec(
        vector_size=config['vector_size'],
        min_count=config['min_count'],
        epochs=config['epochs'],
        window=config['window'],
        dm=config['dm'],
        workers=1,
        seed=RANDOM_STATE,
    )
    model.build_vocab(tagged)
    model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs)
    vectors = np.vstack([model.dv[str(i)] for i in range(len(posts))])
    return vectors


def train_word2vec_bin_embeddings(posts, config):
    sentences = [p['tokens'] or ['empty'] for p in posts]
    w2v = Word2Vec(
        sentences=sentences,
        vector_size=config['vector_size'],
        window=config['window'],
        min_count=config['min_count'],
        workers=1,
        epochs=config['epochs'],
        seed=RANDOM_STATE,
    )

    vocab_words = list(w2v.wv.index_to_key)
    if not vocab_words:
        raise RuntimeError('Word2Vec produced an empty vocabulary.')

    word_vectors = np.vstack([w2v.wv[w] for w in vocab_words])
    word_vectors = normalize_vectors(word_vectors)
    bins = min(config['bins'], len(vocab_words))
    kmeans = KMeans(n_clusters=bins, random_state=RANDOM_STATE, n_init=10)
    word_labels = kmeans.fit_predict(word_vectors)
    word_to_bin = dict(zip(vocab_words, word_labels))

    doc_vectors = np.zeros((len(posts), bins), dtype=np.float32)
    for i, post in enumerate(posts):
        tokens = post['tokens']
        if not tokens:
            continue
        matched = 0
        for token in tokens:
            if token in word_to_bin:
                doc_vectors[i, word_to_bin[token]] += 1.0
                matched += 1
        if matched > 0:
            doc_vectors[i] /= float(matched)
    return doc_vectors, bins


def evaluate_method(method_name, vectors, posts, output_dir, k=5):
    labels, norm_vectors, sil = cosine_cluster(vectors, k)
    intra = average_intra_cluster_similarity(vectors, labels)
    cluster_info = cluster_descriptions(posts, labels)
    image_name = '%s_pca.png' % method_name
    plot_pca(vectors, labels, '%s clustered with cosine-aware KMeans (k=%d)' % (method_name, k), os.path.join(output_dir, image_name))
    result = {
        'method': method_name,
        'n_posts': len(posts),
        'n_clusters': k,
        'vector_dimension': int(np.asarray(vectors).shape[1]),
        'silhouette_cosine': float(round(sil, 4)),
        'avg_intra_cluster_similarity': float(round(intra, 4)),
        'cluster_details': cluster_info,
        'plot': image_name,
    }
    return result


def choose_best(results):
    return sorted(results, key=lambda r: (r['silhouette_cosine'], r['avg_intra_cluster_similarity']), reverse=True)[0]


def method_family(method_name):
    return 'Doc2Vec' if method_name.startswith('doc2vec') else 'Word2Vec+Bins'


def write_report(posts, doc_results, w2v_results, output_dir):
    all_results = doc_results + w2v_results
    best_overall = choose_best(all_results)
    dimensions = sorted(set([r['vector_dimension'] for r in all_results]))

    lines = []
    lines.append('# Lab 8 Comparative Embedding Analysis')
    lines.append('')
    lines.append('## Dataset')
    lines.append('')
    lines.append('- Posts analyzed: %d' % len(posts))
    lines.append('- Source database: `%s`' % DEFAULT_DB)
    lines.append('- Clustering method: KMeans on L2-normalized vectors (cosine-aware approximation) with k = 5')
    lines.append('- Evaluation methods: cosine silhouette score, average intra-cluster similarity, and manual inspection of cluster titles')
    lines.append('')
    lines.append('## Why these evaluation methods?')
    lines.append('')
    lines.append('1. **Cosine silhouette score** checks whether documents are closer to their own cluster than to other clusters when compared by direction rather than raw magnitude.')
    lines.append('2. **Average intra-cluster similarity** rewards embeddings that keep documents near their own centroid.')
    lines.append('3. **Qualitative inspection** is necessary because the assignment asks which embedding best represents document meaning, and that cannot be answered by one numeric score alone.')
    lines.append('')
    lines.append('## Results table')
    lines.append('')
    lines.append('| Method | Family | Dim | Silhouette (cosine) | Avg intra-cluster similarity |')
    lines.append('|---|---|---:|---:|---:|')
    for r in sorted(all_results, key=lambda x: (method_family(x['method']), x['vector_dimension'])):
        lines.append('| %s | %s | %d | %.4f | %.4f |' % (
            r['method'], method_family(r['method']), r['vector_dimension'], r['silhouette_cosine'], r['avg_intra_cluster_similarity']
        ))
    lines.append('')
    lines.append('## Best overall configuration')
    lines.append('')
    lines.append('The strongest configuration in this run was **%s** with cosine silhouette **%.4f** and average intra-cluster similarity **%.4f**.' % (
        best_overall['method'], best_overall['silhouette_cosine'], best_overall['avg_intra_cluster_similarity']
    ))
    lines.append('')
    for dim in dimensions:
        doc_match = [r for r in doc_results if r['vector_dimension'] == dim]
        w2v_match = [r for r in w2v_results if r['vector_dimension'] == dim]
        if doc_match and w2v_match:
            d = doc_match[0]
            w = w2v_match[0]
            winner = d if (d['silhouette_cosine'], d['avg_intra_cluster_similarity']) >= (w['silhouette_cosine'], w['avg_intra_cluster_similarity']) else w
            lines.append('For the **%d-dimensional** comparison, **%s** performed better.' % (dim, winner['method']))
    lines.append('')
    lines.append('## Comparative discussion')
    lines.append('')
    lines.append('### Doc2Vec')
    lines.append('')
    lines.append('- Learns one embedding directly for each full post, so it usually captures document-level context better.')
    lines.append('- Better when titles and bodies talk about the same theme using different vocabulary.')
    lines.append('- More expensive to train and somewhat sensitive to hyperparameters such as vector size and epochs.')
    lines.append('')
    lines.append('### Word2Vec + Bag-of-Words bins')
    lines.append('')
    lines.append('- Easier to explain because each document vector is a normalized count of learned word bins.')
    lines.append('- Preserves some semantic grouping at the word level while still acting like a bag-of-words representation.')
    lines.append('- Loses word order and deeper document context, so clusters can become more about vocabulary frequency than complete meaning.')
    lines.append('')
    lines.append('## Qualitative cluster examples')
    lines.append('')
    for r in [best_overall]:
        lines.append('### %s' % r['method'])
        lines.append('')
        for c in r['cluster_details'][:3]:
            lines.append('- Cluster %d (%d posts): top words = %s' % (c['cluster_id'], c['size'], ', '.join(c['top_words'][:6])))
            for title in c['sample_titles'][:3]:
                lines.append('  - %s' % title.replace('\n', ' ').strip())
        lines.append('')
    lines.append('## Final answer to the assignment questions')
    lines.append('')
    lines.append('Based on the numeric metrics and the sample titles, **%s** was the best representation for this dataset.' % best_overall['method'])
    lines.append('For same-dimension comparisons, the winning method at each dimension is listed above. In general, Doc2Vec tends to represent document meaning more directly, while Word2Vec+Bins is more interpretable but weaker at preserving overall context.')
    lines.append('')

    report_md = os.path.join(output_dir, 'lab8_report.md')
    with open(report_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return report_md


def create_experiment_package(db_path, output_dir, k=5):
    ensure_dir(output_dir)
    posts = load_posts(db_path)
    log.info('Loaded %d posts from %s', len(posts), db_path)

    doc_results = []
    for cfg in DOC2VEC_CONFIGS:
        log.info('Training %s', cfg['name'])
        vectors = train_doc2vec(posts, cfg)
        res = evaluate_method(cfg['name'], vectors, posts, output_dir, k=k)
        doc_results.append(res)
        save_json(os.path.join(output_dir, '%s_results.json' % cfg['name']), res)

    w2v_results = []
    for cfg in WORD2VEC_CONFIGS:
        log.info('Training %s', cfg['name'])
        vectors, bins = train_word2vec_bin_embeddings(posts, cfg)
        res = evaluate_method(cfg['name'], vectors, posts, output_dir, k=k)
        res['word_bins'] = int(bins)
        w2v_results.append(res)
        save_json(os.path.join(output_dir, '%s_results.json' % cfg['name']), res)

    summary = {
        'doc2vec_results': doc_results,
        'word2vec_bin_results': w2v_results,
        'best_overall': choose_best(doc_results + w2v_results),
    }
    save_json(os.path.join(output_dir, 'lab8_summary.json'), summary)
    write_report(posts, doc_results, w2v_results, output_dir)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Lab 8 embedding experiments on the Reddit project database.')
    parser.add_argument('--db', default=DEFAULT_DB, help='SQLite database path')
    parser.add_argument('--out', default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--clusters', type=int, default=5, help='Number of clusters to use for comparison')
    args = parser.parse_args()

    create_experiment_package(args.db, args.out, k=args.clusters)
    print('Lab 8 outputs written to %s' % args.out)
