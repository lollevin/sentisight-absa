"""
Bilingual translations for SentiSight — EN / ZH (Chinese Simplified)
"""

TRANSLATIONS: dict[str, dict[str, str]] = {
    "EN": {
        # ── App identity
        "app_title": "SentiSight",
        "app_subtitle": "AI Marketing Insight Engine",
        "app_desc": "Powered by Aspect-Based Sentiment Analysis (ABSA)  |  Transform customer reviews into actionable marketing intelligence",
        "lang_toggle_label": "Language",

        # ── Sidebar
        "sidebar_title": "🔮 SentiSight",
        "sidebar_api_config": "⚙️ API Configuration",
        "sidebar_provider": "Provider",
        "sidebar_model": "Model",
        "sidebar_api_key": "API Key",
        "sidebar_api_placeholder": "sk-...",
        "sidebar_how_to_get": "📖 How to get API Key",
        "sidebar_deepseek_steps": "**DeepSeek** (Recommended — very cheap):\n1. Go to [platform.deepseek.com](https://platform.deepseek.com)\n2. Register & top up\n3. Create API key",
        "sidebar_openai_steps": "**OpenAI:**\n1. Go to [platform.openai.com](https://platform.openai.com)\n2. Add billing & create API key",
        "sidebar_about": "**📊 About ABSA**",
        "sidebar_about_text": "Aspect-Based Sentiment Analysis identifies sentiment at the dimension level (delivery, quality, packaging...) rather than just overall positive/negative.",

        # ── Problem Statement
        "problem_expander": "📌 Marketing Problem Statement",
        "problem_title": "❌ The Problem",
        "problem_body": (
            "Enterprises struggle to transform large volumes of unstructured customer reviews into usable "
            "marketing insights. Traditional methods — star ratings and manual review — are <b>inefficient "
            "and incomplete</b>. They can only reflect overall satisfaction and cannot identify specific "
            "dimensions of product/service evaluation.<br><br>"
            "<b>Example:</b> A review saying <i>\"delivery was fast but packaging was terrible\"</i> would simply "
            "be labelled <b>\"Neutral\"</b> by traditional tools — providing zero actionable guidance."
        ),
        "solution_title": "✅ The AI Solution",
        "solution_body": (
            "<b>SentiSight</b> uses Aspect-Based Sentiment Analysis (ABSA) powered by large language models "
            "to identify customer attitudes towards <b>specific dimensions</b> of products or services.<br><br>"
            "<b>Same example output:</b><br>"
            "🟢 <b>Delivery</b>: Positive (97% confidence)<br>"
            "🔴 <b>Packaging</b>: Negative (95% confidence)<br>"
            "💡 <b>Action</b>: Highlight delivery speed in ads; improve packaging urgently."
        ),

        # ── Tabs
        "tab1": "🔍  Single Review Analysis",
        "tab2": "📂  Batch Analysis",
        "tab3": "⚡  AI vs Traditional",
        "tab4": "🧪  Testing & Validation",

        # ── Tab 1
        "tab1_title": "#### Analyze a Customer Review",
        "tab1_caption": "Enter any product/service review. The AI will extract aspect-level sentiments and generate marketing recommendations.",
        "tab1_example_label": "💡 Try an example:",
        "tab1_example_placeholder": "(Enter your own review below)",
        "tab1_textarea_placeholder": "e.g., The delivery speed is very fast, but the packaging quality is very poor.",
        "tab1_analyze_btn": "🔍 Analyze",
        "tab1_api_warn": "⚠️ Please enter your API key in the sidebar first.",
        "tab1_api_error": "API key is required. Set it in the sidebar.",
        "tab1_spinner": "🤖 AI is analyzing the review...",
        "tab1_fail": "Analysis failed",
        "tab1_ai_header": "🤖 ABSA — AI Analysis",
        "tab1_overall": "Overall Sentiment:",
        "tab1_no_aspects": "No specific aspects detected in this review.",
        "tab1_suggestions_header": "💡 Marketing Recommendations",
        "tab1_no_suggestions": "No suggestions generated.",
        "tab1_keywords": "**🏷️ Keywords:**",
        "tab1_trad_header": "📉 Traditional VADER",
        "tab1_vader_only": "Overall Label Only",
        "tab1_vader_limit": "⚠️ Cannot identify specific aspects.<br>No marketing action possible.",
        "tab1_vader_blind_spots": "What VADER cannot detect:",
        "tab1_vader_warn_title": "❌ VADER Limitation",
        "tab1_vader_warn_body": "VADER outputs a single sentiment score — it cannot tell you <i>which</i> product dimensions are positive or negative. Marketers cannot act on \"Neutral\" alone.",

        # ── Tab 2
        "tab2_title": "#### Batch Review Analysis",
        "tab2_caption": "Upload a CSV file with a `review` column to analyze multiple reviews at once.",
        "tab2_download_sample": "⬇️ Download Sample CSV",
        "tab2_upload_label": "Upload CSV (must have a 'review' column)",
        "tab2_no_col_error": "CSV must contain a column named 'review'.",
        "tab2_total": "Total reviews:",
        "tab2_api_warn": "⚠️ Set API key in the sidebar to run batch analysis.",
        "tab2_run_btn": "🚀 Run Batch Analysis",
        "tab2_analyzing": "Analyzing {done}/{total} reviews...",
        "tab2_processing": "Processing...",
        "tab2_kpi_total": "Reviews Analyzed",
        "tab2_kpi_pos": "Positive Rate",
        "tab2_kpi_neg": "Negative Rate",
        "tab2_kpi_top": "Top Aspect",
        "tab2_wordcloud": "**☁️ Keyword Cloud**",
        "tab2_download_results": "⬇️ Download Results CSV",

        # ── Tab 3
        "tab3_title": "#### AI ABSA vs Traditional Sentiment Analysis",
        "tab3_caption": "See why aspect-based analysis provides far superior marketing insights compared to traditional methods.",
        "tab3_compare_title": "##### 📋 Capability Comparison",
        "tab3_cap_col0": "Capability",
        "tab3_cap_col1": "VADER (Traditional)",
        "tab3_cap_col2": "SentiSight ABSA (AI)",
        "tab3_capabilities": [
            "Identify overall sentiment",
            "Detect specific product aspects",
            "Distinguish mixed reviews",
            "Generate marketing actions",
            "Support data-driven decisions",
            "Handle complex, nuanced reviews",
            "Scalable to thousands of reviews",
        ],
        "tab3_examples_title": "##### 🔬 Side-by-Side Review Examples",
        "tab3_review_label": "Review {n}:",
        "tab3_ai_header": "🤖 ABSA Result",
        "tab3_suggestions": "**💡 Suggestions:**",
        "tab3_vader_header": "📉 VADER Result",
        "tab3_vader_only_label": "Only output available",
        "tab3_vader_no_action": "Cannot identify delivery vs packaging.<br>No targeted marketing action possible.",
        "tab3_conclusion": "<b>✅ Conclusion</b><br>ABSA provides <b>5x more actionable information</b> per review compared to traditional sentiment analysis. Instead of a single \"Neutral\" label, marketers receive specific aspect-level insights and concrete recommendations — enabling data-driven decisions on advertising, product improvement, and customer retention strategies.",

        # ── Tab 4
        "tab4_methodology_header": "Testing Methodology",
        "tab4_step1_title": "① Benchmark Dataset",
        "tab4_step1_desc": "{n} manually labeled aspect-sentiment pairs as ground truth",
        "tab4_step2_title": "② Run ABSA Model",
        "tab4_step2_desc": "SentiSight analyzes each review and predicts aspect sentiments",
        "tab4_step3_title": "③ Compare Results",
        "tab4_step3_desc": "Predicted aspects matched against ground-truth labels",
        "tab4_step4_title": "④ Score vs VADER",
        "tab4_step4_desc": "Precision / Recall / F1 compared with traditional VADER baseline",
        "tab4_title": "#### Model Testing & Validation",
        "tab4_caption": "Evaluate SentiSight's ABSA accuracy against a manually labeled benchmark dataset of 30 aspect-sentiment pairs. Compare performance against VADER baseline.",
        "tab4_info": "📋 Benchmark dataset: **{pairs}** labeled aspect-sentiment pairs across **{reviews}** unique reviews covering 12+ aspect categories.",
        "tab4_view_data": "📄 View Labeled Test Dataset",
        "tab4_download_test": "⬇️ Download Test Dataset",
        "tab4_api_warn": "⚠️ Enter your API key in the sidebar to run live evaluation.",
        "tab4_run_btn": "🧪 Run Full Evaluation",
        "tab4_spinner": "Running evaluation — this may take ~1 minute...",
        "tab4_analyzing": "Analyzing review {done}/{total}...",
        "tab4_results_title": "##### 📊 Evaluation Results",
        "tab4_kpi_detect": "Aspect Detection Rate",
        "tab4_kpi_acc": "Sentiment Accuracy",
        "tab4_kpi_absa_f1": "Macro F1 Score (ABSA)",
        "tab4_kpi_vader_f1": "Macro F1 Score (VADER)",
        "tab4_metrics_title": "##### 📈 Precision / Recall / F1 by Sentiment Class",
        "tab4_detail_title": "##### 🔍 Per-Aspect Prediction Detail",
        "tab4_summary": "<b>✅ Validation Summary</b><br>SentiSight ABSA correctly detected <b>{detected}/{total}</b> aspects ({detect_rate:.1%} detection rate) with a sentiment accuracy of <b>{acc:.1%}</b> on detected aspects.<br>Macro F1: <b>ABSA = {absa_f1:.2f}</b> vs <b>VADER = {vader_f1:.2f}</b> — demonstrating the clear superiority of aspect-level AI analysis for marketing applications.",
        "tab4_placeholder": "Click <b>Run Full Evaluation</b> above to test the model against the labeled benchmark dataset.",
        "tab4_placeholder_sub": "Results include: aspect detection rate, sentiment accuracy, precision, recall, F1 per class, and comparison with VADER baseline.",
        "tab4_col_sentiment": "Sentiment",
        "tab4_col_absa_p": "ABSA Precision",
        "tab4_col_absa_r": "ABSA Recall",
        "tab4_col_absa_f": "ABSA F1",
        "tab4_col_vader_p": "VADER Precision",
        "tab4_col_vader_r": "VADER Recall",
        "tab4_col_vader_f": "VADER F1",
        "tab4_col_support": "Support",
    },

    "ZH": {
        # ── App identity
        "app_title": "慧感",
        "app_subtitle": "AI 营销洞察引擎",
        "app_desc": "基于方面的情感分析（ABSA）驱动  |  将用户评论转化为可操作的营销情报",
        "lang_toggle_label": "语言",

        # ── Sidebar
        "sidebar_title": "🔮 慧感",
        "sidebar_api_config": "⚙️ API 配置",
        "sidebar_provider": "服务商",
        "sidebar_model": "模型",
        "sidebar_api_key": "API 密钥",
        "sidebar_api_placeholder": "sk-...",
        "sidebar_how_to_get": "📖 如何获取 API 密钥",
        "sidebar_deepseek_steps": "**DeepSeek**（推荐，费用极低）：\n1. 访问 [platform.deepseek.com](https://platform.deepseek.com)\n2. 注册并充值\n3. 创建 API 密钥",
        "sidebar_openai_steps": "**OpenAI：**\n1. 访问 [platform.openai.com](https://platform.openai.com)\n2. 添加付款方式并创建密钥",
        "sidebar_about": "**📊 关于 ABSA**",
        "sidebar_about_text": "基于方面的情感分析可识别具体维度（配送、品质、包装等）的情感倾向，而不只是整体正面/负面。",

        # ── Problem Statement
        "problem_expander": "📌 营销问题陈述",
        "problem_title": "❌ 问题所在",
        "problem_body": (
            "企业难以将大量非结构化用户评论转化为可用的营销洞察。传统方法——星级评分和人工审核——<b>效率低下且信息不完整</b>，"
            "只能反映整体满意度，无法识别产品或服务评价的具体维度。<br><br>"
            "<b>示例：</b>评论 <i>\"配送很快但包装很差\"</i> 在传统工具中只会被标记为 <b>\"中性\"</b>——无法提供任何可操作的指导。"
        ),
        "solution_title": "✅ AI 解决方案",
        "solution_body": (
            "<b>慧感</b>利用大语言模型驱动的基于方面的情感分析（ABSA），识别用户对产品或服务<b>具体维度</b>的态度。<br><br>"
            "<b>同一评论的输出结果：</b><br>"
            "🟢 <b>配送</b>：正面（置信度 97%）<br>"
            "🔴 <b>包装</b>：负面（置信度 95%）<br>"
            "💡 <b>建议</b>：在广告中突出快速配送优势；立即改善包装质量。"
        ),

        # ── Tabs
        "tab1": "🔍  单条评论分析",
        "tab2": "📂  批量分析",
        "tab3": "⚡  AI vs 传统方法",
        "tab4": "🧪  测试与验证",

        # ── Tab 1
        "tab1_title": "#### 分析用户评论",
        "tab1_caption": "输入任意产品或服务评论，AI 将提取各方面情感并生成营销建议。",
        "tab1_example_label": "💡 选择示例：",
        "tab1_example_placeholder": "（在下方输入自定义评论）",
        "tab1_textarea_placeholder": "例如：配送速度非常快，但包装质量很差。",
        "tab1_analyze_btn": "🔍 开始分析",
        "tab1_api_warn": "⚠️ 请先在侧边栏输入 API 密钥。",
        "tab1_api_error": "需要 API 密钥，请在侧边栏设置。",
        "tab1_spinner": "🤖 AI 正在分析评论...",
        "tab1_fail": "分析失败",
        "tab1_ai_header": "🤖 ABSA — AI 分析结果",
        "tab1_overall": "整体情感：",
        "tab1_no_aspects": "未在此评论中检测到具体方面。",
        "tab1_suggestions_header": "💡 营销建议",
        "tab1_no_suggestions": "未生成建议。",
        "tab1_keywords": "**🏷️ 关键词：**",
        "tab1_trad_header": "📉 传统 VADER 结果",
        "tab1_vader_only": "仅能提供整体标签",
        "tab1_vader_limit": "⚠️ 无法识别具体方面。<br>无法生成营销行动建议。",
        "tab1_vader_blind_spots": "VADER 无法检测的方面：",
        "tab1_vader_warn_title": "❌ VADER 的局限性",
        "tab1_vader_warn_body": "VADER 只输出一个整体情感分数——无法告诉你<i>哪个</i>产品维度是正面或负面的。营销人员无法依据单一的「中性」做出决策。",

        # ── Tab 2
        "tab2_title": "#### 批量评论分析",
        "tab2_caption": "上传包含 `review` 列的 CSV 文件，一次性分析多条评论。",
        "tab2_download_sample": "⬇️ 下载示例 CSV",
        "tab2_upload_label": "上传 CSV（必须含 'review' 列）",
        "tab2_no_col_error": "CSV 文件必须包含名为 'review' 的列。",
        "tab2_total": "评论总数：",
        "tab2_api_warn": "⚠️ 请在侧边栏设置 API 密钥后运行批量分析。",
        "tab2_run_btn": "🚀 运行批量分析",
        "tab2_analyzing": "正在分析第 {done}/{total} 条评论...",
        "tab2_processing": "处理中...",
        "tab2_kpi_total": "已分析评论数",
        "tab2_kpi_pos": "正面率",
        "tab2_kpi_neg": "负面率",
        "tab2_kpi_top": "最热方面",
        "tab2_wordcloud": "**☁️ 关键词云**",
        "tab2_download_results": "⬇️ 下载结果 CSV",

        # ── Tab 3
        "tab3_title": "#### AI ABSA vs 传统情感分析",
        "tab3_caption": "了解为什么基于方面的分析比传统方法提供更优越的营销洞察。",
        "tab3_compare_title": "##### 📋 能力对比表",
        "tab3_cap_col0": "能力",
        "tab3_cap_col1": "VADER（传统）",
        "tab3_cap_col2": "慧感 ABSA（AI）",
        "tab3_capabilities": [
            "识别整体情感",
            "检测具体产品方面",
            "区分混合评价",
            "生成营销行动建议",
            "支持数据驱动决策",
            "处理复杂细腻的评论",
            "可扩展至数千条评论",
        ],
        "tab3_examples_title": "##### 🔬 评论案例对比",
        "tab3_review_label": "评论 {n}：",
        "tab3_ai_header": "🤖 ABSA 结果",
        "tab3_suggestions": "**💡 建议：**",
        "tab3_vader_header": "📉 VADER 结果",
        "tab3_vader_only_label": "仅有的输出",
        "tab3_vader_no_action": "无法区分配送与包装。<br>无法生成针对性营销行动。",
        "tab3_conclusion": "<b>✅ 结论</b><br>与传统情感分析相比，ABSA 每条评论提供的信息量<b>多 5 倍</b>。相比单一的「中性」标签，营销人员获得了具体的方面级洞察和明确的建议——为广告投放、产品改进和客户留存提供数据支持。",

        # ── Tab 4
        "tab4_methodology_header": "测试方法",
        "tab4_step1_title": "① 基准数据集",
        "tab4_step1_desc": "{n} 条人工标注的方面-情感对作为标准答案",
        "tab4_step2_title": "② 运行 ABSA 模型",
        "tab4_step2_desc": "慧感分析每条评论并预测各方面的情感倾向",
        "tab4_step3_title": "③ 结果对比",
        "tab4_step3_desc": "将预测方面与人工标注标准答案进行匹配比较",
        "tab4_step4_title": "④ 与 VADER 评分对比",
        "tab4_step4_desc": "计算精确率/召回率/F1，与传统 VADER 基准方法对比",
        "tab4_title": "#### 模型测试与验证",
        "tab4_caption": "基于 30 条人工标注的方面-情感基准数据集，评估慧感 ABSA 的准确率，并与 VADER 基准对比。",
        "tab4_info": "📋 基准数据集：**{pairs}** 条标注方面-情感对，覆盖 **{reviews}** 条独立评论，涉及 12+ 个方面类别。",
        "tab4_view_data": "📄 查看标注测试数据集",
        "tab4_download_test": "⬇️ 下载测试数据集",
        "tab4_api_warn": "⚠️ 请在侧边栏输入 API 密钥以运行实时评估。",
        "tab4_run_btn": "🧪 运行完整评估",
        "tab4_spinner": "正在运行评估，约需 1 分钟...",
        "tab4_analyzing": "正在分析第 {done}/{total} 条评论...",
        "tab4_results_title": "##### 📊 评估结果",
        "tab4_kpi_detect": "方面检测率",
        "tab4_kpi_acc": "情感准确率",
        "tab4_kpi_absa_f1": "宏平均 F1（ABSA）",
        "tab4_kpi_vader_f1": "宏平均 F1（VADER）",
        "tab4_metrics_title": "##### 📈 各情感类别的精确率 / 召回率 / F1",
        "tab4_detail_title": "##### 🔍 逐方面预测详情",
        "tab4_summary": "<b>✅ 验证总结</b><br>慧感 ABSA 在 <b>{detected}/{total}</b> 个方面检测正确（检测率 {detect_rate:.1%}），已检测方面的情感准确率为 <b>{acc:.1%}</b>。<br>宏平均 F1：<b>ABSA = {absa_f1:.2f}</b> vs <b>VADER = {vader_f1:.2f}</b>——清晰证明方面级 AI 分析在营销应用中的显著优势。",
        "tab4_placeholder": "点击上方 <b>运行完整评估</b> 对标注基准数据集进行模型测试。",
        "tab4_placeholder_sub": "结果包括：方面检测率、情感准确率、各类别精确率/召回率/F1，以及与 VADER 基准的对比。",
        "tab4_col_sentiment": "情感类别",
        "tab4_col_absa_p": "ABSA 精确率",
        "tab4_col_absa_r": "ABSA 召回率",
        "tab4_col_absa_f": "ABSA F1",
        "tab4_col_vader_p": "VADER 精确率",
        "tab4_col_vader_r": "VADER 召回率",
        "tab4_col_vader_f": "VADER F1",
        "tab4_col_support": "样本数",
    },
}


def t(key: str) -> str:
    """Get translated string for current session language."""
    import streamlit as st
    lang = st.session_state.get("lang", "EN")
    return TRANSLATIONS.get(lang, TRANSLATIONS["EN"]).get(key, TRANSLATIONS["EN"].get(key, key))


def tlist(key: str) -> list:
    """Get translated list for current session language."""
    import streamlit as st
    lang = st.session_state.get("lang", "EN")
    return TRANSLATIONS.get(lang, TRANSLATIONS["EN"]).get(key, TRANSLATIONS["EN"].get(key, []))
