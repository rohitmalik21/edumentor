"""
EduMentor AI - Custom Gradio Theme and CSS
Professional educational interface styling.
"""

CUSTOM_CSS = """
/* Global Styles */
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

/* Header */
.app-header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    margin-bottom: 20px;
    color: white;
}

.app-header h1 {
    color: white !important;
    font-size: 2.2em !important;
    margin-bottom: 5px !important;
}

.app-header p {
    color: rgba(255,255,255,0.9) !important;
    font-size: 1.1em !important;
}

/* Tab Styling */
.tab-nav button {
    font-size: 1.05em !important;
    font-weight: 600 !important;
    padding: 12px 20px !important;
}

.tab-nav button.selected {
    border-bottom: 3px solid #667eea !important;
    color: #667eea !important;
}

/* Section Cards */
.section-card {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
    background: #fafbfc;
}

/* Status Indicators */
.status-success {
    color: #22c55e;
    font-weight: bold;
}

.status-error {
    color: #ef4444;
    font-weight: bold;
}

/* Metrics Dashboard */
.metric-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 8px;
    padding: 15px;
    text-align: center;
}

.metric-value {
    font-size: 2em;
    font-weight: bold;
    color: #4a5568;
}

.metric-label {
    font-size: 0.9em;
    color: #718096;
}

/* Quiz Styling */
.quiz-question {
    background: #f8fafc;
    border-left: 4px solid #667eea;
    padding: 15px;
    margin: 10px 0;
    border-radius: 0 8px 8px 0;
}

/* Footer */
.app-footer {
    text-align: center;
    padding: 15px;
    color: #718096;
    font-size: 0.9em;
    border-top: 1px solid #e2e8f0;
    margin-top: 20px;
}
"""

APP_TITLE = "EduMentor AI"
APP_DESCRIPTION = "Personalized Learning & Assessment Assistant"
APP_HEADER_HTML = """
<div class="app-header">
    <h1>EduMentor AI</h1>
    <p>Transform your study material into an interactive learning experience</p>
    <p style="font-size: 0.85em; opacity: 0.8;">NLP + Speech | Summarize | Ask | Practice | Evaluate</p>
    <p style="font-size: 0.9em; margin-top: 12px; font-weight: bold;">API Driven Cloud Native Solutions | Assignment 2</p>
    <p style="font-size: 0.85em; opacity: 0.9;">Group: 32 | M.Tech AI - Semester 3</p>
    <table style="margin: 12px auto; border-collapse: collapse; font-size: 0.8em; color: white;">
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.5);">
            <th style="padding: 5px 15px; text-align: left; color: white;">S.No.</th>
            <th style="padding: 5px 15px; text-align: left; color: white;">Name</th>
            <th style="padding: 5px 15px; text-align: left; color: white;">Student ID</th>
        </tr>
        <tr><td style="padding: 4px 15px; color: white;">1</td><td style="padding: 4px 15px; color: white;">Rohit Malik</td><td style="padding: 4px 15px; color: white;">2024AC05988</td></tr>
        <tr><td style="padding: 4px 15px; color: white;">2</td><td style="padding: 4px 15px; color: white;">Suraj Prakash Uniyal</td><td style="padding: 4px 15px; color: white;">2024AD05123</td></tr>
        <tr><td style="padding: 4px 15px; color: white;">3</td><td style="padding: 4px 15px; color: white;">Sudhakar Katam</td><td style="padding: 4px 15px; color: white;">2024AC05889</td></tr>
        <tr><td style="padding: 4px 15px; color: white;">4</td><td style="padding: 4px 15px; color: white;">C S Krishna Chaitanya P</td><td style="padding: 4px 15px; color: white;">2024AD05457</td></tr>
        <tr><td style="padding: 4px 15px; color: white;">5</td><td style="padding: 4px 15px; color: white;">Nikhil Gupta</td><td style="padding: 4px 15px; color: white;">2024AC05640</td></tr>
    </table>
</div>
"""
