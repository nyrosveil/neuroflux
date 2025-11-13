"""
🧠 NeuroFlux's Sentiment Agent
Market sentiment analysis with neural processing and flux monitoring.

Built with love by Nyros Veil 🚀

Analyzes social media, news, and market sentiment using AI.
Provides sentiment scores for tokens with real-time flux adaptation.
Integrates with trading agents for sentiment-based decisions.
"""

import os
import json
import time
import requests
from datetime import datetime, timedelta
from termcolor import cprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import NeuroFlux config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# Output directory for sentiment analysis
OUTPUT_DIR = "src/data/sentiment_agent/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_twitter_sentiment(token_symbol):
    """
    Get Twitter sentiment for a token using API.

    Args:
        token_symbol (str): Token symbol (e.g., 'BTC', 'SOL')

    Returns:
        dict: Sentiment analysis results
    """
    # Placeholder for Twitter API integration
    # In full implementation, this would use Twitter API or sentiment analysis services

    sentiment = {
        'positive': 0.0,
        'negative': 0.0,
        'neutral': 0.0,
        'overall_score': 0.0,  # -1 to 1 scale
        'confidence': 0.0,
        'tweet_count': 0,
        'timeframe': '24h'
    }

    # Simulate sentiment based on token
    base_sentiment = {
        'BTC': 0.2,
        'ETH': 0.1,
        'SOL': 0.3,
        'DOGE': -0.1
    }

    if token_symbol in base_sentiment:
        sentiment['overall_score'] = base_sentiment[token_symbol]
        sentiment['positive'] = max(0, sentiment['overall_score'] + 0.5)
        sentiment['negative'] = max(0, -sentiment['overall_score'] + 0.5)
        sentiment['neutral'] = 1.0 - sentiment['positive'] - sentiment['negative']
        sentiment['confidence'] = 0.8
        sentiment['tweet_count'] = 1000

    return sentiment

def get_news_sentiment(token_symbol):
    """
    Get news sentiment for a token using news APIs.

    Args:
        token_symbol (str): Token symbol

    Returns:
        dict: News sentiment analysis
    """
    # Placeholder for news API integration
    news_sentiment = {
        'positive': 0.0,
        'negative': 0.0,
        'neutral': 0.0,
        'overall_score': 0.0,
        'article_count': 0,
        'top_headlines': []
    }

    # Simulate news sentiment
    news_sentiment['overall_score'] = 0.1
    news_sentiment['positive'] = 0.4
    news_sentiment['negative'] = 0.3
    news_sentiment['neutral'] = 0.3
    news_sentiment['article_count'] = 25
    news_sentiment['top_headlines'] = [
        "Bitcoin ETF approval rumors boost market confidence",
        "Ethereum upgrade shows strong network development",
        "Market analysis: Bullish signals emerging"
    ]

    return news_sentiment

def get_reddit_sentiment(token_symbol):
    """
    Get Reddit sentiment for a token.

    Args:
        token_symbol (str): Token symbol

    Returns:
        dict: Reddit sentiment analysis
    """
    # Placeholder for Reddit API integration
    reddit_sentiment = {
        'positive': 0.0,
        'negative': 0.0,
        'neutral': 0.0,
        'overall_score': 0.0,
        'post_count': 0,
        'subreddits': ['r/cryptocurrency', 'r/bitcoin']
    }

    # Simulate Reddit sentiment
    reddit_sentiment['overall_score'] = 0.15
    reddit_sentiment['positive'] = 0.45
    reddit_sentiment['negative'] = 0.25
    reddit_sentiment['neutral'] = 0.3
    reddit_sentiment['post_count'] = 150

    return reddit_sentiment

def calculate_flux_impact(sentiment_score, flux_level):
    """
    Adjust sentiment score based on market flux.

    Args:
        sentiment_score (float): Raw sentiment score
        flux_level (float): Current market flux level

    Returns:
        float: Flux-adjusted sentiment score
    """
    # In high flux, sentiment becomes less reliable
    adjustment_factor = 1.0 - (flux_level * 0.3)  # Reduce influence by up to 30%
    return sentiment_score * adjustment_factor

def aggregate_sentiment(token_symbol, flux_level):
    """
    Aggregate sentiment from multiple sources with neuro-processing.

    Args:
        token_symbol (str): Token symbol
        flux_level (float): Current market flux level

    Returns:
        dict: Aggregated sentiment analysis
    """
    # Get sentiment from different sources
    twitter = get_twitter_sentiment(token_symbol)
    news = get_news_sentiment(token_symbol)
    reddit = get_reddit_sentiment(token_symbol)

    # Apply flux adjustment
    twitter['overall_score'] = calculate_flux_impact(twitter['overall_score'], flux_level)
    news['overall_score'] = calculate_flux_impact(news['overall_score'], flux_level)
    reddit['overall_score'] = calculate_flux_impact(reddit['overall_score'], flux_level)

    # Weighted aggregation (Twitter 40%, News 35%, Reddit 25%)
    weights = {'twitter': 0.4, 'news': 0.35, 'reddit': 0.25}

    overall_score = (
        twitter['overall_score'] * weights['twitter'] +
        news['overall_score'] * weights['news'] +
        reddit['overall_score'] * weights['reddit']
    )

    # Calculate confidence based on data volume and consistency
    data_points = twitter['tweet_count'] + news['article_count'] + reddit['post_count']
    confidence = min(0.9, data_points / 1000)  # Max confidence at 1000+ data points

    # Determine sentiment category
    if overall_score > 0.1:
        category = 'bullish'
        color = 'green'
    elif overall_score < -0.1:
        category = 'bearish'
        color = 'red'
    else:
        category = 'neutral'
        color = 'yellow'

    aggregated = {
        'token': token_symbol,
        'timestamp': datetime.now().isoformat(),
        'overall_score': round(overall_score, 3),
        'category': category,
        'confidence': round(confidence, 3),
        'flux_level': flux_level,
        'sources': {
            'twitter': twitter,
            'news': news,
            'reddit': reddit
        },
        'weights': weights,
        'recommendation': get_sentiment_recommendation(overall_score, confidence, flux_level)
    }

    return aggregated

def get_sentiment_recommendation(score, confidence, flux_level):
    """
    Generate trading recommendation based on sentiment analysis.

    Args:
        score (float): Overall sentiment score
        confidence (float): Analysis confidence
        flux_level (float): Market flux level

    Returns:
        str: Trading recommendation
    """
    if confidence < 0.5:
        return "Low confidence - wait for more data"

    if flux_level > FLUX_SENSITIVITY:
        return "High flux detected - sentiment less reliable, reduce position sizes"

    if score > 0.2:
        return "Strong bullish sentiment - consider long positions"
    elif score > 0.1:
        return "Moderately bullish - monitor for entry"
    elif score < -0.2:
        return "Strong bearish sentiment - consider short positions"
    elif score < -0.1:
        return "Moderately bearish - monitor for exit"
    else:
        return "Neutral sentiment - hold current positions"

def save_sentiment_analysis(results):
    """
    Save sentiment analysis results.

    Args:
        results (dict): Sentiment analysis results
    """
    # Save latest analysis
    with open(f"{OUTPUT_DIR}/latest_sentiment.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Append to historical data
    with open(f"{OUTPUT_DIR}/sentiment_history.jsonl", 'a') as f:
        f.write(json.dumps(results, default=str) + '\n')

    # Update CSV summary
    update_sentiment_csv(results)

def update_sentiment_csv(results):
    """
    Update the sentiment summary CSV.

    Args:
        results (dict): Sentiment results
    """
    csv_path = f"{OUTPUT_DIR}/sentiment_summary.csv"

    # Check if file exists
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as f:
        if not file_exists:
            f.write("timestamp,token,overall_score,category,confidence,flux_level,recommendation\\n")

        f.write(f"{results['timestamp']},{results['token']},{results['overall_score']},{results['category']},{results['confidence']},{results['flux_level']},\"{results['recommendation']}\"\\n")

def display_sentiment(results):
    """
    Display sentiment analysis results in a formatted way.

    Args:
        results (dict): Sentiment analysis results
    """
    cprint(f"\\n📊 Sentiment Analysis for {results['token']}", "cyan", attrs=['bold'])
    cprint(f"Overall Score: {results['overall_score']:.3f}", "white")
    cprint(f"Category: {results['category']}", "white")
    cprint(f"Confidence: {results['confidence']:.3f}", "white")
    cprint(f"Flux Level: {results['flux_level']:.3f}", "white")

    # Color code the score
    if results['category'] == 'bullish':
        cprint(f"🟢 {results['recommendation']}", "green")
    elif results['category'] == 'bearish':
        cprint(f"🔴 {results['recommendation']}", "red")
    else:
        cprint(f"🟡 {results['recommendation']}", "yellow")

    # Show source breakdown
    cprint("\\n📈 Source Breakdown:", "blue")
    for source, data in results['sources'].items():
        score = data['overall_score']
        color = 'green' if score > 0 else 'red' if score < 0 else 'white'
        cprint(f"  {source.title()}: {score:.3f}", color)

def main():
    """Main sentiment analysis loop with neuro-flux processing."""
    cprint("🧠 NeuroFlux Sentiment Agent starting...", "cyan")
    cprint("Market sentiment analysis with neural processing", "yellow")

    while True:
        try:
            # Get tokens to analyze
            tokens = get_active_tokens()
            if not tokens:
                tokens = ['BTC', 'ETH', 'SOL']  # Default tokens for testing

            cprint(f"📊 Analyzing sentiment for {len(tokens)} tokens", "blue")

            for token in tokens:
                # Calculate current flux level
                flux_level = FLUX_SENSITIVITY  # Placeholder - integrate with flux monitoring

                # Perform sentiment analysis
                results = aggregate_sentiment(token, flux_level)

                # Display results
                display_sentiment(results)

                # Save results
                save_sentiment_analysis(results)

                # Brief pause between tokens
                time.sleep(2)

            cprint(f"✅ Sentiment analysis cycle complete - sleeping {SLEEP_BETWEEN_RUNS_MINUTES} minutes", "green")
            time.sleep(SLEEP_BETWEEN_RUNS_MINUTES * 60)

        except KeyboardInterrupt:
            cprint("\\n👋 Sentiment Agent stopped by user", "yellow")
            break
        except Exception as e:
            cprint(f"❌ Sentiment Agent error: {str(e)}", "red")
            time.sleep(60)  # Brief pause on error

if __name__ == "__main__":
    main()