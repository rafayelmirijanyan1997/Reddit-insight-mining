from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from pathlib import Path

ROOT = Path(__file__).resolve().parent

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleBlue', parent=styles['Title'], alignment=TA_CENTER, textColor=colors.HexColor('#173f73'), fontSize=22, leading=28, spaceAfter=18))
styles.add(ParagraphStyle(name='HeadingBlue', parent=styles['Heading2'], textColor=colors.HexColor('#173f73'), spaceBefore=10, spaceAfter=8))
styles.add(ParagraphStyle(name='BodySoft', parent=styles['BodyText'], fontSize=10.5, leading=15, spaceAfter=7))
styles.add(ParagraphStyle(name='SmallSoft', parent=styles['BodyText'], fontSize=9, leading=12, spaceAfter=5))


def build_readme_pdf():
    doc = SimpleDocTemplate(str(ROOT / 'README.pdf'), pagesize=letter,
                            leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.8 * inch, bottomMargin=0.7 * inch)
    story = []
    story.append(Paragraph('Reddit Insight Mining and Clustering<br/>Lab 8 Submission Guide', styles['TitleBlue']))
    story.append(Paragraph('This PDF summarizes the finished project package, the Lab 8 extension, and how to run the full embedding comparison workflow.', styles['BodySoft']))

    sections = [
        ('What is included', [
            'Original Reddit data pipeline: scrape, clean, enrich, vectorize, cluster, and visualize posts.',
            'Lab 8 extension in <b>lab8_pipeline.py</b> for 3 Doc2Vec configurations and 3 Word2Vec+bin configurations.',
            'Python 3.6-compatible type hint fixes for course environments that reject built-in generics like list[str].',
            'Submission documents: this README PDF and a meeting notes PDF template.',
        ]),
        ('Main Lab 8 command', [
            'Run from the project directory:',
            '<font name="Courier">python lab8_pipeline.py --db reddit_posts.db --out lab8_outputs --clusters 5</font>',
            'The command writes PCA plots, per-method JSON metrics, a summary JSON file, and a written Markdown analysis.',
        ]),
        ('How the comparison works', [
            'Doc2Vec is trained at 50, 100, and 200 dimensions.',
            'Word2Vec is trained at 50, 100, and 200 dimensions, then vocabulary vectors are clustered into semantic bins.',
            'Each document becomes a normalized bag-of-bins vector for the Word2Vec-based method.',
            'All document vectors are L2-normalized before KMeans so the clustering behaves like a cosine-aware approximation.',
            'Methods are compared with cosine silhouette score, average intra-cluster similarity, and manual cluster inspection.',
        ]),
        ('Recommended submission checklist', [
            'Verify the team name and member names in the meeting notes PDF.',
            'Run the Lab 8 pipeline on the final database used by your team.',
            'Review the generated report and plots in lab8_outputs before recording the demo video.',
            'Zip the entire project folder after outputs are generated.',
        ]),
    ]

    for heading, bullets in sections:
        story.append(Paragraph(heading, styles['HeadingBlue']))
        for item in bullets:
            story.append(Paragraph('&bull; ' + item, styles['BodySoft']))

    story.append(Spacer(1, 0.15 * inch))
    data = [
        ['File', 'Purpose'],
        ['lab8_pipeline.py', 'Runs all embedding experiments and writes comparison outputs'],
        ['requirements.txt', 'Lists Python package dependencies'],
        ['readme.md', 'Detailed project overview and commands'],
        ['README.pdf', 'Submission-friendly run guide'],
        ['meeting_notes_L8_team_name.pdf', 'Template for team meeting minutes'],
    ]
    table = Table(data, colWidths=[2.1 * inch, 3.9 * inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#173f73')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#c6d3e1')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7f9fc')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(Paragraph('Included deliverables', styles['HeadingBlue']))
    story.append(table)
    doc.build(story)


def build_meeting_notes_pdf():
    doc = SimpleDocTemplate(str(ROOT / 'meeting_notes_L8_team_name.pdf'), pagesize=letter,
                            leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.8 * inch, bottomMargin=0.7 * inch)
    story = []
    story.append(Paragraph('Meeting Notes - Lab 8', styles['TitleBlue']))
    story.append(Paragraph('Replace the placeholders below with your actual team name, team members, and meeting details before submission.', styles['BodySoft']))

    meta = [
        ['Team name', 'team_name'],
        ['Members', 'Member 1, Member 2, Member 3'],
        ['Assignment', 'DSCI 560 Lab 8 - Representing Document Concepts with Embeddings'],
    ]
    meta_table = Table(meta, colWidths=[1.4 * inch, 4.6 * inch])
    meta_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#c6d3e1')),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#eef3f9')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.18 * inch))

    meetings = [
        ('Meeting 1', 'Date: ____________', 'Focus: review Lab 8 requirements, divide Doc2Vec and Word2Vec tasks, confirm database and preprocessing status.'),
        ('Meeting 2', 'Date: ____________', 'Focus: run initial Doc2Vec experiments, inspect cluster outputs, document hyperparameter choices.'),
        ('Meeting 3', 'Date: ____________', 'Focus: implement Word2Vec word-binning approach and compare dimension-matched results.'),
        ('Meeting 4', 'Date: ____________', 'Focus: finalize evaluation, write comparative analysis, and prepare the demo video plan.'),
        ('Meeting 5', 'Date: ____________', 'Focus: package code, verify PDFs, and complete final submission QA.'),
    ]
    for title, date_line, focus in meetings:
        story.append(Paragraph(title, styles['HeadingBlue']))
        rows = [
            ['Date', Paragraph(date_line.replace('Date: ', ''), styles['BodySoft'])],
            ['Attendees', Paragraph('_______________________________________________', styles['BodySoft'])],
            ['Discussion summary', Paragraph(focus, styles['SmallSoft'])],
            ['Decisions / next steps', Paragraph('_______________________________________________________________', styles['BodySoft'])],
        ]
        tbl = Table(rows, colWidths=[1.75 * inch, 4.25 * inch])
        tbl.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.45, colors.HexColor('#ccd7e5')),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f4f7fb')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.14 * inch))

    doc.build(story)


if __name__ == '__main__':
    build_readme_pdf()
    build_meeting_notes_pdf()
