"""
🧠 NeuroFlux's Web Search Agent
Web search with neuro-filtering.

Built with love by Nyros Veil 🚀

Performs intelligent web searches and filters results using AI.
Finds relevant market information, news, and trading insights.
"""

import os
import time
import json
import requests
from datetime import datetime
from termcolor import cprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import NeuroFlux config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Import ModelFactory for AI filtering
from models.model_factory import ModelFactory

# Output directory for web search results
OUTPUT_DIR = "src/data/websearch_agent/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Search engines and APIs to use
SEARCH_ENGINES = [
    'https://api.duckduckgo.com/',
    # Add more search APIs as needed
]

def perform_web_search(query, max_results=10):
    """
    Perform web search using available APIs.

    Args:
        query (str): Search query
        max_results (int): Maximum number of results to return

    Returns:
        list: List of search results
    """
    results = []

    try:
        # DuckDuckGo Instant Answer API (free)
        ddg_url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': 1,
            'skip_disambig': 1
        }

        response = requests.get(ddg_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extract search results
        if data.get('Results'):
            for result in data['Results'][:max_results]:
                results.append({
                    'title': result.get('Text', ''),
                    'url': result.get('FirstURL', ''),
                    'snippet': result.get('Text', ''),
                    'source': 'DuckDuckGo'
                })

        # If no Results, try RelatedTopics
        if not results and data.get('RelatedTopics'):
            for topic in data['RelatedTopics'][:max_results]:
                if isinstance(topic, dict) and 'Text' in topic:
                    results.append({
                        'title': topic.get('Text', '').split(' - ')[0],
                        'url': topic.get('FirstURL', ''),
                        'snippet': topic.get('Text', ''),
                        'source': 'DuckDuckGo'
                    })

        # Fallback mock results if API fails
        if not results:
            results = [
                {
                    'title': f'Mock result for: {query}',
                    'url': 'https://example.com',
                    'snippet': f'This is a mock search result for the query: {query}',
                    'source': 'Mock'
                }
            ]

        return results[:max_results]

    except Exception as e:
        cprint(f"❌ Web search error: {str(e)}", "red")
        return []

def filter_results_with_ai(search_results, original_query):
    """
    Filter and rank search results using AI analysis.

    Args:
        search_results (list): Raw search results
        original_query (str): Original search query

    Returns:
        dict: Filtered and analyzed results
    """
    try:
        # Initialize AI model
        model = ModelFactory.create_model('anthropic')

        # Prepare results for AI analysis
        results_text = ""
        for i, result in enumerate(search_results[:5]):  # Analyze top 5 results
            results_text += f"""
            Result {i+1}:
            Title: {result['title']}
            URL: {result['url']}
            Snippet: {result['snippet']}
            """

        filter_prompt = f"""
        Analyze these web search results for the query: "{original_query}"

        Search Results:
        {results_text}

        Your task is to:
        1. Filter out irrelevant or low-quality results
        2. Rank the remaining results by relevance and credibility
        3. Extract key insights from the most relevant results
        4. Assess overall market sentiment from the results
        5. Identify any breaking news or important developments

        Provide analysis in JSON format with these keys:
        relevant_results (array of indices of relevant results),
        top_result_index (index of most relevant result),
        key_insights (array of key insights found),
        market_sentiment (BULLISH, BEARISH, NEUTRAL, or UNKNOWN),
        breaking_news (any breaking news found, or empty string),
        credibility_score (1-10 overall credibility of results)
        """

        response = model.generate_response("", filter_prompt, temperature=0.2, max_tokens=1000)

        # Parse JSON response
        try:
            analysis = json.loads(response.strip())
        except json.JSONDecodeError:
            # Fallback analysis
            analysis = {
                'relevant_results': list(range(len(search_results))),
                'top_result_index': 0,
                'key_insights': ['AI analysis inconclusive'],
                'market_sentiment': 'NEUTRAL',
                'breaking_news': '',
                'credibility_score': 5
            }

        # Apply filtering
        filtered_results = []
        relevant_indices = analysis.get('relevant_results', [])

        for i, result in enumerate(search_results):
            if i in relevant_indices:
                result_copy = result.copy()
                result_copy['relevance_score'] = len(relevant_indices) - relevant_indices.index(i)  # Higher score for earlier in list
                result_copy['is_top_result'] = (i == analysis.get('top_result_index', -1))
                filtered_results.append(result_copy)

        analysis['filtered_results'] = filtered_results

        return analysis

    except Exception as e:
        cprint(f"❌ AI filtering error: {str(e)}", "red")
        return {
            'relevant_results': list(range(len(search_results))),
            'top_result_index': 0,
            'key_insights': ['Filtering failed'],
            'market_sentiment': 'UNKNOWN',
            'breaking_news': '',
            'credibility_score': 1,
            'filtered_results': search_results
        }

def generate_search_topics():
    """
    Generate relevant search topics based on current market conditions.

    Returns:
        list: List of search queries to perform
    """
    # Base topics for market research
    base_topics = [
        "cryptocurrency market news today",
        "stock market outlook 2024",
        "federal reserve interest rate decision",
        "bitcoin price prediction",
        "economic indicators latest",
        "geopolitical events affecting markets",
        "commodity prices update",
        "trading volume analysis"
    ]

    # Add token-specific searches if configured
    if hasattr(config, 'MONITORED_TOKENS') and config.MONITORED_TOKENS:
        for token in config.MONITORED_TOKENS[:3]:  # Limit to 3 tokens
            base_topics.append(f"{token} price analysis")
            base_topics.append(f"{token} news developments")

    return base_topics

def save_search_results(query, raw_results, analysis):
    """
    Save search results and analysis.

    Args:
        query (str): Search query
        raw_results (list): Raw search results
        analysis (dict): AI analysis results
    """
    result = {
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'raw_results_count': len(raw_results),
        'filtered_results_count': len(analysis.get('filtered_results', [])),
        'analysis': analysis,
        'raw_results': raw_results
    }

    # Save latest search
    with open(f"{OUTPUT_DIR}/latest_search.json", 'w') as f:
        json.dump(result, f, indent=2, default=str)

    # Append to history
    with open(f"{OUTPUT_DIR}/search_history.jsonl", 'a') as f:
        f.write(json.dumps(result, default=str) + '\n')

def main():
    """Main web search and filtering loop."""
    cprint("🧠 NeuroFlux Web Search Agent starting...", "cyan")
    cprint("Performing intelligent web searches with neuro-filtering", "yellow")

    while True:
        try:
            # Generate search topics
            search_topics = generate_search_topics()
            cprint(f"🔍 Generated {len(search_topics)} search topics", "blue")

            total_insights = 0
            sentiment_summary = {'BULLISH': 0, 'BEARISH': 0, 'NEUTRAL': 0, 'UNKNOWN': 0}

            # Perform searches for each topic
            for query in search_topics:
                cprint(f"🌐 Searching: {query}", "white")

                # Perform web search
                raw_results = perform_web_search(query, max_results=8)

                if raw_results:
                    cprint(f"📄 Found {len(raw_results)} results", "blue")

                    # Filter with AI
                    analysis = filter_results_with_ai(raw_results, query)

                    # Update sentiment tracking
                    sentiment = analysis.get('market_sentiment', 'UNKNOWN')
                    sentiment_summary[sentiment] = sentiment_summary.get(sentiment, 0) + 1

                    # Count insights
                    insights = analysis.get('key_insights', [])
                    total_insights += len(insights)

                    cprint(f"🎯 Filtered to {len(analysis.get('filtered_results', []))} relevant results", "green")
                    cprint(f"📊 Sentiment: {sentiment} | Insights: {len(insights)}", "yellow")

                    # Save results
                    save_search_results(query, raw_results, analysis)
                else:
                    cprint("❌ No search results found", "red")

                # Brief pause between searches (respect rate limits)
                time.sleep(2)

            # Summary
            dominant_sentiment = max(sentiment_summary.items(), key=lambda x: x[1])[0]
            cprint(f"📈 Session Summary: {total_insights} insights | Dominant Sentiment: {dominant_sentiment}", "green")

            cprint(f"✅ Web search cycle complete - sleeping {config.SLEEP_BETWEEN_RUNS_MINUTES} minutes", "green")
            time.sleep(config.SLEEP_BETWEEN_RUNS_MINUTES * 60)

        except KeyboardInterrupt:
            cprint("\n🛑 Web Search Agent stopped by user", "yellow")
            break
        except Exception as e:
            cprint(f"❌ Web Search Agent error: {str(e)}", "red")
            time.sleep(60)  # Brief pause on error

if __name__ == "__main__":
    main()