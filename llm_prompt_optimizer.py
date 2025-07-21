#!/usr/bin/env python3
"""
LLM Prompt Optimization Tool
A comprehensive tool for optimizing prompts for Large Language Models.
Features prompt analysis, A/B testing, optimization suggestions, and performance tracking.
"""

import json
import re
import time
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
from collections import Counter
import argparse


class PromptType(Enum):
    """Types of prompts for different use cases."""
    INSTRUCTION = "instruction"
    QUESTION_ANSWER = "question_answer"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"
    CODE_GENERATION = "code_generation"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"


class OptimizationStrategy(Enum):
    """Different optimization strategies."""
    CLARITY = "clarity"
    SPECIFICITY = "specificity"
    CONTEXT = "context"
    STRUCTURE = "structure"
    EXAMPLES = "examples"
    CONSTRAINTS = "constraints"
    PERSONA = "persona"
    CHAIN_OF_THOUGHT = "chain_of_thought"


@dataclass
class PromptMetrics:
    """Metrics for evaluating prompt quality."""
    clarity_score: float = 0.0
    specificity_score: float = 0.0
    length_score: float = 0.0
    complexity_score: float = 0.0
    structure_score: float = 0.0
    example_score: float = 0.0
    constraint_score: float = 0.0
    overall_score: float = 0.0


@dataclass
class OptimizationSuggestion:
    """A specific optimization suggestion for a prompt."""
    strategy: OptimizationStrategy
    priority: str  # high, medium, low
    description: str
    example: str
    impact_estimate: float  # 0.0 to 1.0


@dataclass
class PromptVariant:
    """A variant of a prompt for A/B testing."""
    id: str
    content: str
    strategy_applied: List[OptimizationStrategy]
    metrics: PromptMetrics
    test_results: Dict[str, Any]


@dataclass
class PromptTest:
    """Results from testing a prompt variant."""
    variant_id: str
    response_quality: float
    response_time: float
    token_count: int
    coherence_score: float
    relevance_score: float
    completeness_score: float
    timestamp: float


class PromptAnalyzer:
    """Core analyzer for evaluating prompt quality."""
    
    def __init__(self):
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'you', 'your', 'i', 'me', 'my'
        }
        
        # Patterns for different prompt elements
        self.patterns = {
            'instruction_words': r'\b(please|generate|create|write|analyze|explain|describe|list|identify|compare|summarize)\b',
            'question_words': r'\b(what|how|why|when|where|who|which|can|could|would|should|do|does|did)\b',
            'constraint_words': r'\b(must|should|cannot|don\'t|avoid|exclude|limit|restrict|only|exactly)\b',
            'example_markers': r'\b(for example|such as|like|including|e\.g\.|i\.e\.)\b',
            'context_markers': r'\b(given|considering|assuming|in the context of|background|scenario)\b',
            'output_format': r'\b(format|structure|template|json|csv|list|bullet|numbered)\b'
        }
    
    def analyze_prompt(self, prompt: str, prompt_type: Optional[PromptType] = None) -> PromptMetrics:
        """Analyze a prompt and return quality metrics."""
        metrics = PromptMetrics()
        
        # Basic preprocessing
        prompt_lower = prompt.lower().strip()
        words = prompt_lower.split()
        sentences = [s.strip() for s in prompt.split('.') if s.strip()]
        
        # Clarity Score (0-1)
        metrics.clarity_score = self._calculate_clarity(prompt, words, sentences)
        
        # Specificity Score (0-1)
        metrics.specificity_score = self._calculate_specificity(prompt_lower, words)
        
        # Length Score (0-1) - optimal length considerations
        metrics.length_score = self._calculate_length_score(len(words))
        
        # Complexity Score (0-1)
        metrics.complexity_score = self._calculate_complexity(sentences, words)
        
        # Structure Score (0-1)
        metrics.structure_score = self._calculate_structure(prompt, sentences)
        
        # Example Score (0-1)
        metrics.example_score = self._calculate_example_usage(prompt_lower)
        
        # Constraint Score (0-1)
        metrics.constraint_score = self._calculate_constraints(prompt_lower)
        
        # Overall Score (weighted average)
        weights = {
            'clarity': 0.25,
            'specificity': 0.20,
            'structure': 0.15,
            'examples': 0.15,
            'constraints': 0.10,
            'length': 0.10,
            'complexity': 0.05
        }
        
        metrics.overall_score = (
            metrics.clarity_score * weights['clarity'] +
            metrics.specificity_score * weights['specificity'] +
            metrics.structure_score * weights['structure'] +
            metrics.example_score * weights['examples'] +
            metrics.constraint_score * weights['constraints'] +
            metrics.length_score * weights['length'] +
            metrics.complexity_score * weights['complexity']
        )
        
        return metrics
    
    def _calculate_clarity(self, prompt: str, words: List[str], sentences: List[str]) -> float:
        """Calculate clarity score based on language patterns."""
        score = 0.0
        
        # Check for clear instructions
        instruction_matches = len(re.findall(self.patterns['instruction_words'], prompt, re.IGNORECASE))
        if instruction_matches > 0:
            score += 0.3
        
        # Check for question clarity
        question_matches = len(re.findall(self.patterns['question_words'], prompt, re.IGNORECASE))
        if question_matches > 0:
            score += 0.2
        
        # Penalize for ambiguous words
        ambiguous_words = ['thing', 'stuff', 'something', 'anything', 'everything']
        ambiguous_count = sum(1 for word in words if word in ambiguous_words)
        if ambiguous_count == 0:
            score += 0.2
        else:
            score = max(0, score - (ambiguous_count * 0.1))
        
        # Check sentence structure
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if 10 <= avg_sentence_length <= 25:  # Optimal range
                score += 0.3
            elif avg_sentence_length > 35:
                score -= 0.1
        
        return min(1.0, score)
    
    def _calculate_specificity(self, prompt_lower: str, words: List[str]) -> float:
        """Calculate specificity score."""
        score = 0.0
        
        # Check for specific domain terms (simple heuristic)
        specific_indicators = ['specific', 'detailed', 'exact', 'precise', 'particular']
        if any(indicator in prompt_lower for indicator in specific_indicators):
            score += 0.2
        
        # Check for numbers/quantities
        if re.search(r'\d+', prompt_lower):
            score += 0.2
        
        # Check for proper nouns (capitalized words)
        proper_nouns = sum(1 for word in words if word.istitle())
        if proper_nouns > 0:
            score += min(0.3, proper_nouns * 0.1)
        
        # Check for technical terms or domain-specific language
        technical_patterns = r'\b(API|JSON|CSV|SQL|HTML|algorithm|model|dataset|parameters)\b'
        if re.search(technical_patterns, prompt_lower):
            score += 0.3
        
        return min(1.0, score)
    
    def _calculate_length_score(self, word_count: int) -> float:
        """Calculate optimal length score."""
        # Optimal range: 20-150 words
        if 20 <= word_count <= 150:
            return 1.0
        elif word_count < 20:
            return word_count / 20.0
        elif word_count > 150:
            return max(0.3, 1.0 - (word_count - 150) / 200.0)
        return 0.5
    
    def _calculate_complexity(self, sentences: List[str], words: List[str]) -> float:
        """Calculate complexity score (simpler is often better for prompts)."""
        if not sentences:
            return 0.0
        
        # Average sentence length
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Vocabulary diversity (unique words / total words)
        unique_words = set(word.lower() for word in words if word.lower() not in self.stop_words)
        vocab_diversity = len(unique_words) / max(1, len(words))
        
        # Optimal complexity: not too simple, not too complex
        length_score = 1.0 - abs(avg_sentence_length - 15) / 30.0  # Optimal around 15 words
        diversity_score = min(1.0, vocab_diversity * 2)  # Encourage vocabulary diversity
        
        return max(0.0, (length_score + diversity_score) / 2)
    
    def _calculate_structure(self, prompt: str, sentences: List[str]) -> float:
        """Calculate structure score."""
        score = 0.0
        
        # Check for clear sections/organization
        if '\n' in prompt:
            score += 0.2
        
        # Check for bullet points or numbered lists
        if re.search(r'^\s*[\-\*\d+\.]', prompt, re.MULTILINE):
            score += 0.3
        
        # Check for output format specifications
        if re.search(self.patterns['output_format'], prompt, re.IGNORECASE):
            score += 0.3
        
        # Check for logical flow (context -> task -> constraints)
        context_found = re.search(self.patterns['context_markers'], prompt, re.IGNORECASE)
        instruction_found = re.search(self.patterns['instruction_words'], prompt, re.IGNORECASE)
        constraint_found = re.search(self.patterns['constraint_words'], prompt, re.IGNORECASE)
        
        if context_found and instruction_found:
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_example_usage(self, prompt_lower: str) -> float:
        """Calculate score for example usage."""
        score = 0.0
        
        # Check for example markers
        example_matches = len(re.findall(self.patterns['example_markers'], prompt_lower))
        if example_matches > 0:
            score += 0.5
        
        # Check for actual examples (quoted text or code blocks)
        if '"' in prompt_lower or "'" in prompt_lower:
            score += 0.3
        
        if '```' in prompt_lower or '`' in prompt_lower:
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_constraints(self, prompt_lower: str) -> float:
        """Calculate score for clear constraints."""
        constraint_matches = len(re.findall(self.patterns['constraint_words'], prompt_lower))
        return min(1.0, constraint_matches * 0.25)


class PromptOptimizer:
    """Main optimizer class for generating improved prompt variants."""
    
    def __init__(self):
        self.analyzer = PromptAnalyzer()
    
    def generate_suggestions(self, prompt: str, prompt_type: Optional[PromptType] = None) -> List[OptimizationSuggestion]:
        """Generate optimization suggestions for a prompt."""
        metrics = self.analyzer.analyze_prompt(prompt, prompt_type)
        suggestions = []
        
        # Clarity improvements
        if metrics.clarity_score < 0.7:
            suggestions.append(OptimizationSuggestion(
                strategy=OptimizationStrategy.CLARITY,
                priority="high",
                description="Improve clarity by using more specific action words and reducing ambiguity",
                example="Replace 'analyze this' with 'analyze the sentiment of this customer review'",
                impact_estimate=0.8
            ))
        
        # Specificity improvements
        if metrics.specificity_score < 0.6:
            suggestions.append(OptimizationSuggestion(
                strategy=OptimizationStrategy.SPECIFICITY,
                priority="high",
                description="Add specific requirements, constraints, or domain context",
                example="Add: 'Focus on technical accuracy and provide code examples'",
                impact_estimate=0.7
            ))
        
        # Structure improvements
        if metrics.structure_score < 0.5:
            suggestions.append(OptimizationSuggestion(
                strategy=OptimizationStrategy.STRUCTURE,
                priority="medium",
                description="Organize the prompt with clear sections: Context, Task, Output Format",
                example="Structure as: Background: [context]\nTask: [instruction]\nFormat: [requirements]",
                impact_estimate=0.6
            ))
        
        # Example improvements
        if metrics.example_score < 0.4:
            suggestions.append(OptimizationSuggestion(
                strategy=OptimizationStrategy.EXAMPLES,
                priority="medium",
                description="Include concrete examples to clarify expectations",
                example="Add: 'For example: Input: \"Great product!\" → Output: {\"sentiment\": \"positive\", \"confidence\": 0.9}'",
                impact_estimate=0.6
            ))
        
        # Chain of thought for complex tasks
        if prompt_type in [PromptType.ANALYTICAL, PromptType.CODE_GENERATION] and 'step' not in prompt.lower():
            suggestions.append(OptimizationSuggestion(
                strategy=OptimizationStrategy.CHAIN_OF_THOUGHT,
                priority="medium",
                description="Add chain-of-thought reasoning for complex tasks",
                example="Add: 'Think through this step-by-step: 1) First analyze... 2) Then consider... 3) Finally conclude...'",
                impact_estimate=0.7
            ))
        
        # Constraint improvements
        if metrics.constraint_score < 0.3:
            suggestions.append(OptimizationSuggestion(
                strategy=OptimizationStrategy.CONSTRAINTS,
                priority="low",
                description="Add clear constraints and limitations",
                example="Add: 'Limit response to 200 words. Do not include speculative information.'",
                impact_estimate=0.4
            ))
        
        # Persona for specific domains
        if prompt_type == PromptType.CREATIVE:
            suggestions.append(OptimizationSuggestion(
                strategy=OptimizationStrategy.PERSONA,
                priority="low",
                description="Add a persona to guide the response style",
                example="Add: 'Act as an experienced creative director with 10 years in advertising'",
                impact_estimate=0.5
            ))
        
        return sorted(suggestions, key=lambda x: x.impact_estimate, reverse=True)
    
    def create_optimized_variants(self, original_prompt: str, suggestions: List[OptimizationSuggestion], max_variants: int = 3) -> List[PromptVariant]:
        """Create optimized prompt variants based on suggestions."""
        variants = []
        
        # Original as baseline
        original_metrics = self.analyzer.analyze_prompt(original_prompt)
        variants.append(PromptVariant(
            id="original",
            content=original_prompt,
            strategy_applied=[],
            metrics=original_metrics,
            test_results={}
        ))
        
        # High-impact suggestions first
        high_impact_suggestions = [s for s in suggestions if s.impact_estimate > 0.6]
        
        for i, suggestion in enumerate(high_impact_suggestions[:max_variants]):
            optimized_prompt = self._apply_optimization(original_prompt, suggestion)
            optimized_metrics = self.analyzer.analyze_prompt(optimized_prompt)
            
            variants.append(PromptVariant(
                id=f"variant_{i+1}",
                content=optimized_prompt,
                strategy_applied=[suggestion.strategy],
                metrics=optimized_metrics,
                test_results={}
            ))
        
        # Combined optimization variant
        if len(high_impact_suggestions) > 1:
            combined_prompt = original_prompt
            applied_strategies = []
            
            for suggestion in high_impact_suggestions[:2]:  # Combine top 2
                combined_prompt = self._apply_optimization(combined_prompt, suggestion)
                applied_strategies.append(suggestion.strategy)
            
            combined_metrics = self.analyzer.analyze_prompt(combined_prompt)
            variants.append(PromptVariant(
                id="combined",
                content=combined_prompt,
                strategy_applied=applied_strategies,
                metrics=combined_metrics,
                test_results={}
            ))
        
        return variants
    
    def _apply_optimization(self, prompt: str, suggestion: OptimizationSuggestion) -> str:
        """Apply a specific optimization to a prompt."""
        # This is a simplified version - in practice, you'd have more sophisticated
        # prompt transformation logic for each strategy
        
        if suggestion.strategy == OptimizationStrategy.CLARITY:
            if not re.search(r'\b(please|generate|create|write|analyze)\b', prompt, re.IGNORECASE):
                prompt = "Please " + prompt.lower()
            
        elif suggestion.strategy == OptimizationStrategy.STRUCTURE:
            if '\n' not in prompt:
                # Add basic structure
                prompt = f"Task: {prompt}\n\nRequirements:\n- Be specific and accurate\n- Provide clear examples"
        
        elif suggestion.strategy == OptimizationStrategy.EXAMPLES:
            if 'example' not in prompt.lower():
                prompt += "\n\nFor example: [Provide a concrete example here]"
        
        elif suggestion.strategy == OptimizationStrategy.CHAIN_OF_THOUGHT:
            if 'step' not in prompt.lower():
                prompt += "\n\nPlease think through this step-by-step before providing your final answer."
        
        elif suggestion.strategy == OptimizationStrategy.CONSTRAINTS:
            prompt += "\n\nConstraints: Keep the response concise and factual."
        
        elif suggestion.strategy == OptimizationStrategy.SPECIFICITY:
            if not re.search(r'\d+', prompt):
                prompt = prompt.replace("some", "3-5").replace("many", "multiple")
        
        return prompt


class PromptTester:
    """A/B testing framework for prompt variants."""
    
    def __init__(self):
        self.test_results = []
    
    def simulate_test(self, variant: PromptVariant) -> PromptTest:
        """Simulate testing a prompt variant (in real implementation, this would call an LLM API)."""
        # Simulate based on metrics - higher quality prompts get better results
        base_quality = variant.metrics.overall_score
        
        # Add some random variation
        import random
        quality_variation = random.uniform(-0.1, 0.1)
        response_quality = max(0.0, min(1.0, base_quality + quality_variation))
        
        # Simulate response time (better prompts often get faster responses)
        response_time = random.uniform(1.0, 5.0) * (1.1 - base_quality)
        
        # Simulate token count (more specific prompts might generate longer responses)
        token_count = int(random.uniform(50, 500) * (0.5 + variant.metrics.specificity_score))
        
        # Simulate other metrics
        coherence = max(0.0, min(1.0, response_quality + random.uniform(-0.05, 0.05)))
        relevance = max(0.0, min(1.0, response_quality + random.uniform(-0.1, 0.1)))
        completeness = max(0.0, min(1.0, response_quality + random.uniform(-0.15, 0.05)))
        
        return PromptTest(
            variant_id=variant.id,
            response_quality=response_quality,
            response_time=response_time,
            token_count=token_count,
            coherence_score=coherence,
            relevance_score=relevance,
            completeness_score=completeness,
            timestamp=time.time()
        )
    
    def run_ab_test(self, variants: List[PromptVariant], num_tests: int = 5) -> Dict[str, List[PromptTest]]:
        """Run A/B tests on multiple prompt variants."""
        results = {}
        
        for variant in variants:
            results[variant.id] = []
            for _ in range(num_tests):
                test_result = self.simulate_test(variant)
                results[variant.id].append(test_result)
                self.test_results.append(test_result)
        
        return results
    
    def analyze_test_results(self, results: Dict[str, List[PromptTest]]) -> Dict[str, Dict[str, float]]:
        """Analyze A/B test results and provide statistical summary."""
        analysis = {}
        
        for variant_id, tests in results.items():
            if tests:
                analysis[variant_id] = {
                    'avg_quality': statistics.mean(t.response_quality for t in tests),
                    'avg_response_time': statistics.mean(t.response_time for t in tests),
                    'avg_token_count': statistics.mean(t.token_count for t in tests),
                    'avg_coherence': statistics.mean(t.coherence_score for t in tests),
                    'avg_relevance': statistics.mean(t.relevance_score for t in tests),
                    'avg_completeness': statistics.mean(t.completeness_score for t in tests),
                    'quality_std': statistics.stdev(t.response_quality for t in tests) if len(tests) > 1 else 0,
                    'consistency': 1.0 - (statistics.stdev(t.response_quality for t in tests) if len(tests) > 1 else 0)
                }
        
        return analysis


def main():
    """CLI interface for the prompt optimizer."""
    parser = argparse.ArgumentParser(description='LLM Prompt Optimization Tool')
    parser.add_argument('prompt', help='Prompt to optimize (or file path)')
    parser.add_argument('--type', choices=[t.value for t in PromptType], 
                       help='Type of prompt')
    parser.add_argument('--output', '-o', help='Output file for results')
    parser.add_argument('--format', choices=['json', 'text'], default='text',
                       help='Output format')
    parser.add_argument('--test', action='store_true',
                       help='Run A/B testing on variants')
    parser.add_argument('--variants', type=int, default=3,
                       help='Number of variants to generate')
    
    args = parser.parse_args()
    
    # Load prompt
    if args.prompt.endswith('.txt'):
        with open(args.prompt, 'r') as f:
            prompt = f.read().strip()
    else:
        prompt = args.prompt
    
    prompt_type = PromptType(args.type) if args.type else None
    
    # Initialize optimizer
    optimizer = PromptOptimizer()
    
    # Analyze original prompt
    print("Analyzing original prompt...")
    metrics = optimizer.analyzer.analyze_prompt(prompt, prompt_type)
    
    # Generate suggestions
    suggestions = optimizer.generate_suggestions(prompt, prompt_type)
    
    # Create variants
    variants = optimizer.create_optimized_variants(prompt, suggestions, args.variants)
    
    # Run tests if requested
    test_results = None
    test_analysis = None
    if args.test:
        print("Running A/B tests...")
        tester = PromptTester()
        test_results = tester.run_ab_test(variants)
        test_analysis = tester.analyze_test_results(test_results)
    
    # Generate report
    if args.format == 'json':
        report = {
            'original_prompt': prompt,
            'original_metrics': asdict(metrics),
            'suggestions': [asdict(s) for s in suggestions],
            'variants': [asdict(v) for v in variants],
            'test_results': test_analysis
        }
        output = json.dumps(report, indent=2)
    else:
        output = generate_text_report(prompt, metrics, suggestions, variants, test_analysis)
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Results written to {args.output}")
    else:
        print(output)


def generate_text_report(prompt: str, metrics: PromptMetrics, suggestions: List[OptimizationSuggestion], 
                        variants: List[PromptVariant], test_analysis: Optional[Dict] = None) -> str:
    """Generate a human-readable text report."""
    lines = []
    lines.append("=" * 80)
    lines.append("LLM PROMPT OPTIMIZATION REPORT")
    lines.append("=" * 80)
    
    # Original prompt metrics
    lines.append(f"\nORIGINAL PROMPT ANALYSIS:")
    lines.append(f"Overall Score: {metrics.overall_score:.2f}")
    lines.append(f"  Clarity: {metrics.clarity_score:.2f}")
    lines.append(f"  Specificity: {metrics.specificity_score:.2f}")
    lines.append(f"  Structure: {metrics.structure_score:.2f}")
    lines.append(f"  Examples: {metrics.example_score:.2f}")
    lines.append(f"  Constraints: {metrics.constraint_score:.2f}")
    
    # Suggestions
    if suggestions:
        lines.append(f"\nOPTIMIZATION SUGGESTIONS:")
        for i, suggestion in enumerate(suggestions, 1):
            lines.append(f"\n{i}. {suggestion.strategy.value.title()} [{suggestion.priority.upper()}]")
            lines.append(f"   {suggestion.description}")
            lines.append(f"   Example: {suggestion.example}")
            lines.append(f"   Impact: {suggestion.impact_estimate:.1%}")
    
    # Variants
    lines.append(f"\nGENERATED VARIANTS:")
    for variant in variants[1:]:  # Skip original
        lines.append(f"\n{variant.id.upper()}:")
        lines.append(f"Score: {variant.metrics.overall_score:.2f} (vs {metrics.overall_score:.2f} original)")
        lines.append(f"Strategies: {', '.join(s.value for s in variant.strategy_applied)}")
        lines.append(f"Content: {variant.content[:100]}...")
    
    # Test results
    if test_analysis:
        lines.append(f"\nA/B TEST RESULTS:")
        for variant_id, results in test_analysis.items():
            lines.append(f"\n{variant_id.upper()}:")
            lines.append(f"  Avg Quality: {results['avg_quality']:.3f}")
            lines.append(f"  Consistency: {results['consistency']:.3f}")
            lines.append(f"  Response Time: {results['avg_response_time']:.2f}s")
    
    return "\n".join(lines)


if __name__ == '__main__':
    main()