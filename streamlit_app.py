#!/usr/bin/env python3
"""
Streamlit Web Interface for LLM Prompt Optimization Tool
A professional interface for analyzing, optimizing, and A/B testing LLM prompts.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from typing import Dict, List
import datetime

# Import our optimizer (assumes the optimizer code is in llm_prompt_optimizer.py)
try:
    from llm_prompt_optimizer import (
        PromptOptimizer, PromptAnalyzer, PromptTester, PromptType, 
        OptimizationStrategy, PromptMetrics, OptimizationSuggestion,
        PromptVariant, PromptTest
    )
except ImportError:
    st.error("Please ensure llm_prompt_optimizer.py is in the same directory")
    st.stop()


def setup_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="LLM Prompt Optimizer",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            margin-bottom: 2rem;
            text-align: center;
        }
        .metric-container {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #007bff;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .suggestion-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-left: 4px solid #28a745;
        }
        .suggestion-high {
            border-left-color: #dc3545;
        }
        .suggestion-medium {
            border-left-color: #ffc107;
        }
        .suggestion-low {
            border-left-color: #28a745;
        }
        .variant-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            border: 2px solid #e9ecef;
        }
        .variant-best {
            border-color: #28a745;
            background: #f8fff8;
        }
        .score-excellent {
            color: #28a745;
            font-weight: bold;
        }
        .score-good {
            color: #ffc107;
            font-weight: bold;
        }
        .score-poor {
            color: #dc3545;
            font-weight: bold;
        }
        .stTextArea textarea {
            font-family: 'Courier New', monospace;
            font-size: 14px;
        }
        .prompt-preview {
            background: #f1f3f4;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #4285f4;
            margin: 1rem 0;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
        }
        </style>
    """, unsafe_allow_html=True)


def get_score_color_class(score: float) -> str:
    """Get CSS class based on score."""
    if score >= 0.8:
        return "score-excellent"
    elif score >= 0.6:
        return "score-good"
    else:
        return "score-poor"


def create_metrics_radar_chart(metrics: PromptMetrics, title: str = "Prompt Quality Metrics") -> go.Figure:
    """Create a radar chart for prompt metrics."""
    categories = [
        'Clarity', 'Specificity', 'Structure', 
        'Examples', 'Constraints', 'Length', 'Complexity'
    ]
    
    values = [
        metrics.clarity_score,
        metrics.specificity_score,
        metrics.structure_score,
        metrics.example_score,
        metrics.constraint_score,
        metrics.length_score,
        metrics.complexity_score
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=title,
        line_color='#667eea',
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title=title,
        height=400
    )
    
    return fig


def create_comparison_chart(variants: List[PromptVariant]) -> go.Figure:
    """Create a comparison chart for prompt variants."""
    if not variants:
        return go.Figure()
    
    variant_names = [v.id for v in variants]
    overall_scores = [v.metrics.overall_score for v in variants]
    clarity_scores = [v.metrics.clarity_score for v in variants]
    specificity_scores = [v.metrics.specificity_score for v in variants]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Overall Score',
        x=variant_names,
        y=overall_scores,
        marker_color='#667eea'
    ))
    
    fig.add_trace(go.Bar(
        name='Clarity',
        x=variant_names,
        y=clarity_scores,
        marker_color='#764ba2'
    ))
    
    fig.add_trace(go.Bar(
        name='Specificity',
        x=variant_names,
        y=specificity_scores,
        marker_color='#f093fb'
    ))
    
    fig.update_layout(
        title='Prompt Variant Comparison',
        xaxis_title='Variants',
        yaxis_title='Score',
        barmode='group',
        height=400
    )
    
    return fig


def create_ab_test_chart(test_analysis: Dict) -> go.Figure:
    """Create A/B test results visualization."""
    if not test_analysis:
        return go.Figure()
    
    variants = list(test_analysis.keys())
    quality_scores = [test_analysis[v]['avg_quality'] for v in variants]
    consistency_scores = [test_analysis[v]['consistency'] for v in variants]
    response_times = [test_analysis[v]['avg_response_time'] for v in variants]
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Response Quality', 'Consistency', 'Response Time'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Quality scores
    fig.add_trace(
        go.Bar(x=variants, y=quality_scores, name='Quality', marker_color='#667eea'),
        row=1, col=1
    )
    
    # Consistency scores
    fig.add_trace(
        go.Bar(x=variants, y=consistency_scores, name='Consistency', marker_color='#764ba2'),
        row=1, col=2
    )
    
    # Response times
    fig.add_trace(
        go.Bar(x=variants, y=response_times, name='Response Time', marker_color='#f093fb'),
        row=1, col=3
    )
    
    fig.update_layout(
        title_text="A/B Test Results Comparison",
        height=400,
        showlegend=False
    )
    
    return fig


def display_suggestion_cards(suggestions: List[OptimizationSuggestion]):
    """Display optimization suggestions as cards."""
    if not suggestions:
        st.info("🎉 Your prompt looks great! No major optimizations needed.")
        return
    
    for i, suggestion in enumerate(suggestions):
        priority_class = f"suggestion-{suggestion.priority}"
        
        with st.container():
            st.markdown(f"""
                <div class="suggestion-card {priority_class}">
                    <h4>💡 {suggestion.strategy.value.replace('_', ' ').title()}</h4>
                    <p><strong>Priority:</strong> {suggestion.priority.title()} | 
                       <strong>Impact:</strong> {suggestion.impact_estimate:.1%}</p>
                    <p><strong>Suggestion:</strong> {suggestion.description}</p>
                    <p><strong>Example:</strong> <em>{suggestion.example}</em></p>
                </div>
            """, unsafe_allow_html=True)


def display_variant_cards(variants: List[PromptVariant]):
    """Display prompt variants as expandable cards."""
    if len(variants) <= 1:
        return
    
    # Find best variant
    best_variant = max(variants[1:], key=lambda v: v.metrics.overall_score)
    
    for variant in variants:
        is_best = variant.id == best_variant.id and variant.id != "original"
        card_class = "variant-best" if is_best else ""
        
        with st.expander(f"📝 {variant.id.replace('_', ' ').title()} {' (Best!)' if is_best else ''}", expanded=variant.id == "original"):
            st.markdown(f'<div class="variant-card {card_class}">', unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                score_class = get_score_color_class(variant.metrics.overall_score)
                st.markdown(f'<p class="{score_class}">Overall: {variant.metrics.overall_score:.2f}</p>', unsafe_allow_html=True)
            with col2:
                st.write(f"Clarity: {variant.metrics.clarity_score:.2f}")
            with col3:
                st.write(f"Specificity: {variant.metrics.specificity_score:.2f}")
            
            # Strategies applied
            if variant.strategy_applied:
                strategies = ", ".join(s.value.replace('_', ' ').title() for s in variant.strategy_applied)
                st.write(f"**Strategies Applied:** {strategies}")
            
            # Prompt content
            st.markdown(f'<div class="prompt-preview">{variant.content}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)


def get_sample_prompts() -> Dict[str, str]:
    """Get sample prompts for different use cases."""
    return {
        "Basic Question": "What is machine learning?",
        
        "Code Generation": "Write a function to sort a list",
        
        "Analysis Task": "Analyze this data and tell me what you think",
        
        "Creative Writing": "Write a story about AI",
        
        "Well-Structured Prompt": """You are an expert data scientist with 10 years of experience.

Task: Analyze the provided sales data and identify key trends.

Requirements:
- Focus on seasonal patterns and growth trends
- Provide specific metrics and percentages
- Include actionable recommendations

Output Format:
- Executive summary (2-3 sentences)
- Key findings (bullet points)
- Recommendations (numbered list)

Data: [Sales data would be provided here]

Please think through this analysis step-by-step before providing your final recommendations."""
    }


def main():
    """Main Streamlit application."""
    setup_page()
    
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🎯 LLM Prompt Optimizer</h1>
            <p>Professional tool for analyzing, optimizing, and A/B testing LLM prompts</p>
            <p><em>Perfect for ChatGPT, Claude, GPT-4, and other language models</em></p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = PromptOptimizer()
        st.session_state.tester = PromptTester()
        st.session_state.analysis_results = None
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Prompt type selection
        prompt_type_str = st.selectbox(
            "Prompt Type:",
            ["auto-detect"] + [t.value.replace('_', ' ').title() for t in PromptType],
            help="Select the type of prompt for targeted optimization"
        )
        
        prompt_type = None
        if prompt_type_str != "auto-detect":
            prompt_type = PromptType(prompt_type_str.lower().replace(' ', '_'))
        
        # Number of variants
        num_variants = st.slider(
            "Number of Variants:",
            min_value=1,
            max_value=5,
            value=3,
            help="How many optimized variants to generate"
        )
        
        # A/B testing options
        st.subheader("🧪 A/B Testing")
        run_ab_test = st.checkbox(
            "Run A/B Testing",
            help="Simulate testing variants against each other"
        )
        
        if run_ab_test:
            num_tests = st.slider(
                "Tests per Variant:",
                min_value=3,
                max_value=20,
                value=5,
                help="Number of simulated tests to run"
            )
        
        # Export options
        st.subheader("📊 Export")
        if st.session_state.analysis_results:
            if st.button("📄 Export Analysis", use_container_width=True):
                # Create export data
                export_data = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'analysis': st.session_state.analysis_results
                }
                
                json_str = json.dumps(export_data, indent=2, default=str)
                st.download_button(
                    label="Download JSON Report",
                    data=json_str,
                    file_name=f"prompt_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # Sample prompts
        st.subheader("📚 Sample Prompts")
        sample_prompts = get_sample_prompts()
        selected_sample = st.selectbox(
            "Load Sample:",
            ["None"] + list(sample_prompts.keys())
        )
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyze", "📊 Results", "🧪 A/B Testing", "📈 Insights"])
    
    with tab1:
        st.header("Prompt Analysis")
        
        # Prompt input
        prompt_content = ""
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Enter Your Prompt")
        with col2:
            if selected_sample != "None":
                if st.button("Load Sample", use_container_width=True):
                    prompt_content = sample_prompts[selected_sample]
        
        prompt_content = st.text_area(
            "Prompt Content:",
            value=prompt_content,
            height=200,
            placeholder="Enter the prompt you want to optimize...",
            help="Paste your LLM prompt here for analysis and optimization"
        )
        
        # Analysis button
        if prompt_content and st.button("🎯 Analyze & Optimize", type="primary", use_container_width=True):
            with st.spinner("Analyzing prompt and generating optimizations..."):
                # Analyze original prompt
                metrics = st.session_state.optimizer.analyzer.analyze_prompt(prompt_content, prompt_type)
                
                # Generate suggestions
                suggestions = st.session_state.optimizer.generate_suggestions(prompt_content, prompt_type)
                
                # Create variants
                variants = st.session_state.optimizer.create_optimized_variants(
                    prompt_content, suggestions, num_variants
                )
                
                # Run A/B testing if enabled
                test_results = None
                test_analysis = None
                num_tests = 5  # Default value
                if run_ab_test:
                    # num_tests is set by sidebar slider if run_ab_test is True
                    test_results = st.session_state.tester.run_ab_test(variants, num_tests)
                    test_analysis = st.session_state.tester.analyze_test_results(test_results)
                
                # Store results
                st.session_state.analysis_results = {
                    'original_prompt': prompt_content,
                    'prompt_type': prompt_type.value if prompt_type else 'auto-detect',
                    'metrics': metrics,
                    'suggestions': suggestions,
                    'variants': variants,
                    'test_results': test_results,
                    'test_analysis': test_analysis
                }
            
            st.success("✅ Analysis complete! Check the Results tab.")
            st.rerun()
        
        # Show current metrics if available
        if st.session_state.analysis_results:
            st.subheader("Quick Metrics")
            results = st.session_state.analysis_results
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                score = results['metrics'].overall_score
                st.metric("Overall Score", f"{score:.2f}", f"{score:.1%}")
            with col2:
                st.metric("Suggestions", len(results['suggestions']))
            with col3:
                st.metric("Variants", len(results['variants']) - 1)  # Exclude original
            with col4:
                if results['test_analysis']:
                    best_variant = max(results['test_analysis'].items(), 
                                     key=lambda x: x[1]['avg_quality'])
                    st.metric("Best Variant", best_variant[0])
    
    with tab2:
        st.header("Analysis Results")
        
        if not st.session_state.analysis_results:
            st.info("👈 Analyze a prompt in the 'Analyze' tab to see results here.")
        else:
            results = st.session_state.analysis_results
            
            # Original prompt metrics
            st.subheader("📋 Original Prompt Analysis")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                # Metrics display
                metrics = results['metrics']
                st.markdown(f"""
                    <div class="metric-container">
                        <h4>Quality Metrics</h4>
                        <p><strong>Overall Score:</strong> <span class="{get_score_color_class(metrics.overall_score)}">{metrics.overall_score:.2f}</span></p>
                        <p>Clarity: {metrics.clarity_score:.2f}</p>
                        <p>Specificity: {metrics.specificity_score:.2f}</p>
                        <p>Structure: {metrics.structure_score:.2f}</p>
                        <p>Examples: {metrics.example_score:.2f}</p>
                        <p>Constraints: {metrics.constraint_score:.2f}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Radar chart
                fig_radar = create_metrics_radar_chart(metrics, "Original Prompt Metrics")
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # Optimization suggestions
            st.subheader("💡 Optimization Suggestions")
            display_suggestion_cards(results['suggestions'])
            
            # Prompt variants
            st.subheader("📝 Generated Variants")
            display_variant_cards(results['variants'])
            
            # Variant comparison chart
            if len(results['variants']) > 1:
                st.subheader("📊 Variant Comparison")
                fig_comparison = create_comparison_chart(results['variants'])
                st.plotly_chart(fig_comparison, use_container_width=True)
    
    with tab3:
        st.header("A/B Testing Results")
        
        if not st.session_state.analysis_results or not st.session_state.analysis_results['test_analysis']:
            st.info("🧪 Enable A/B Testing in the sidebar and analyze a prompt to see results here.")
        else:
            test_analysis = st.session_state.analysis_results['test_analysis']
            
            # Summary metrics
            st.subheader("📊 Test Summary")
            
            # Find best performing variant
            best_variant = max(test_analysis.items(), key=lambda x: x[1]['avg_quality'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Best Variant", best_variant[0])
            with col2:
                st.metric("Best Quality", f"{best_variant[1]['avg_quality']:.3f}")
            with col3:
                st.metric("Consistency", f"{best_variant[1]['consistency']:.3f}")
            with col4:
                st.metric("Response Time", f"{best_variant[1]['avg_response_time']:.2f}s")
            
            # Detailed results chart
            fig_ab = create_ab_test_chart(test_analysis)
            st.plotly_chart(fig_ab, use_container_width=True)
            
            # Detailed results table
            st.subheader("📈 Detailed Results")
            
            df_data = []
            for variant_id, results in test_analysis.items():
                df_data.append({
                    'Variant': variant_id,
                    'Avg Quality': f"{results['avg_quality']:.3f}",
                    'Consistency': f"{results['consistency']:.3f}",
                    'Response Time': f"{results['avg_response_time']:.2f}s",
                    'Coherence': f"{results['avg_coherence']:.3f}",
                    'Relevance': f"{results['avg_relevance']:.3f}",
                    'Completeness': f"{results['avg_completeness']:.3f}"
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
    
    with tab4:
        st.header("Optimization Insights")
        
        if not st.session_state.analysis_results:
            st.info("📈 Analyze a prompt to see insights and recommendations here.")
        else:
            results = st.session_state.analysis_results
            metrics = results['metrics']
            suggestions = results['suggestions']
            
            # Key insights
            st.subheader("🔍 Key Insights")
            
            # Identify main issues
            weak_areas = []
            if metrics.clarity_score < 0.7:
                weak_areas.append("Clarity")
            if metrics.specificity_score < 0.6:
                weak_areas.append("Specificity")
            if metrics.structure_score < 0.5:
                weak_areas.append("Structure")
            if metrics.example_score < 0.4:
                weak_areas.append("Examples")
            
            if weak_areas:
                st.warning(f"**Areas for Improvement:** {', '.join(weak_areas)}")
            else:
                st.success("**Great job!** Your prompt shows strong quality across all metrics.")
            
            # Priority recommendations
            high_priority = [s for s in suggestions if s.priority == "high"]
            if high_priority:
                st.error("**🚨 High Priority Actions:**")
                for suggestion in high_priority:
                    st.markdown(f"• {suggestion.description}")
            
            # Best practices
            st.subheader("✅ Best Practices Applied")
            
            practices = []
            if metrics.clarity_score >= 0.7:
                practices.append("Clear instructions and language")
            if metrics.specificity_score >= 0.6:
                practices.append("Specific requirements and context")
            if metrics.structure_score >= 0.5:
                practices.append("Well-organized structure")
            if metrics.example_score >= 0.4:
                practices.append("Good use of examples")
            if metrics.constraint_score >= 0.3:
                practices.append("Clear constraints and limitations")
            
            for practice in practices:
                st.success(f"✅ {practice}")
            
            # Industry insights
            st.subheader("🏭 Industry Insights")
            
            st.info("""
            **💡 Prompt Engineering Best Practices:**
            
            • **Clarity First**: Clear, unambiguous instructions lead to 40% better responses
            • **Context Matters**: Providing relevant context improves accuracy by 30%
            • **Examples Work**: Including examples increases task completion by 50%
            • **Structure Helps**: Well-organized prompts reduce confusion and improve consistency
            • **Test Iteratively**: A/B testing variants can improve performance by 25%
            """)
            
            # Performance predictions
            if results['test_analysis']:
                st.subheader("🎯 Performance Predictions")
                
                best_variant = max(results['test_analysis'].items(), 
                                 key=lambda x: x[1]['avg_quality'])
                improvement = (best_variant[1]['avg_quality'] - 
                             results['test_analysis']['original']['avg_quality']) * 100
                
                if improvement > 0:
                    st.success(f"**Expected Improvement:** +{improvement:.1f}% in response quality")
                    st.info(f"**Recommended Variant:** {best_variant[0]} shows the best performance")
                else:
                    st.info("Your original prompt is already well-optimized!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem;">
            <h4>🎯 LLM Prompt Optimizer</h4>
            <p>Built with expertise in prompt engineering and adversarial testing</p>
            <p>Perfect for optimizing prompts for ChatGPT, Claude, GPT-4, and other LLMs</p>
            <p><em>Professional tool for AI practitioners and prompt engineers</em></p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()