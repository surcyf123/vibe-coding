import os
import json
import time
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("btc_sentiment_trader")

# API Keys
DATURA_API_KEY = os.getenv("DATURA_API_KEY")
CHUTES_API_KEY = os.getenv("CHUTES_API_KEY")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

# Trading parameters
SYMBOL = "BTCUSDT"
BASE_POSITION_SIZE = 1.0  # USDT
MAX_LEVERAGE = 100
SENTIMENT_THRESHOLD = 20  # Minimum sentiment score to trigger a trade

# Initialize Binance client
binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

async def http_request_with_retry(session: aiohttp.ClientSession, method: str, url: str, **kwargs):
    """
    Make an HTTP request with retry logic.
    """
    for attempt in range(3):
        try:
            async with session.request(method, url, timeout=30, **kwargs) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Request failed (attempt {attempt+1}/3): {str(e)}")
            if attempt < 3 - 1:
                await asyncio.sleep(1 * (attempt + 1))
            else:
                raise

async def get_twitter_data():
    """
    Fetch recent tweets about Bitcoin using Datura API
    """
    url = "https://apis.datura.ai/twitter"
    headers = {
        "Authorization": DATURA_API_KEY,
        "Content-Type": "application/json"
    }
    
    # Calculate date range (last 24 hours)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    payload = {
        "query": "Bitcoin OR BTC price OR crypto market",
        "blue_verified": True,  # Focus on verified accounts for more reliable signals
        "end_date": end_date,
        "start_date": start_date,
        "lang": "en",
        "min_likes": 50,  # Only tweets with some engagement
        "min_replies": 5,
        "min_retweets": 10,
        "sort": "Top",
        "count": 20  # Get top 20 tweets
    }
    
    logger.info("Fetching tweets from Datura API")
    async with aiohttp.ClientSession() as session:
        try:
            data = await http_request_with_retry(session, "POST", url, headers=headers, json=payload)
            logger.info(f"Retrieved {len(data)} tweets")
            return data
        except Exception as e:
            logger.error(f"Error fetching tweets: {str(e)}")
            return []

async def analyze_sentiment(tweets):
    """
    Analyze the sentiment of tweets using Chutes LLM
    """
    if not tweets:
        logger.warning("No tweets to analyze")
        return 0
    
    # Prepare tweets for analysis
    tweet_texts = "\n".join([f"Tweet {i+1}: {tweet['text']}" for i, tweet in enumerate(tweets[:10])])
    
    prompt = f"""
    Analyze the sentiment of these Bitcoin-related tweets and provide a single sentiment score from -100 (extremely bearish) to +100 (extremely bullish).
    
    {tweet_texts}
    
    Consider factors like:
    - Price predictions
    - Market analysis
    - Trader sentiment
    - News impact
    
    Return only a single integer number between -100 and 100 representing the overall sentiment.
    """
    
    logger.info("Analyzing sentiment with Chutes LLM")
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://llm.chutes.ai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {CHUTES_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": "unsloth/gemma-3-4b-it",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "max_tokens": 50,
                "temperature": 0.3
            }
            
            data = await http_request_with_retry(session, "POST", url, headers=headers, json=payload)
            sentiment_text = data['choices'][0]['message']['content'].strip()
            
            # Extract the numeric sentiment score
            try:
                sentiment_score = int(''.join(filter(lambda x: x.isdigit() or x == '-', sentiment_text)))
                # Ensure the score is within bounds
                sentiment_score = max(-100, min(100, sentiment_score))
                logger.info(f"Sentiment analysis complete. Score: {sentiment_score}")
                return sentiment_score
            except ValueError:
                logger.error(f"Failed to parse sentiment score from: {sentiment_text}")
                return 0
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        return 0

def calculate_position_size_and_side(sentiment_score):
    """
    Calculate position size and side based on sentiment score
    """
    # Determine trade direction
    if sentiment_score > 0:
        side = "BUY"  # Long
    elif sentiment_score < 0:
        side = "SELL"  # Short
    else:
        return None, None, 0
    
    # Calculate leverage based on sentiment magnitude
    leverage = abs(sentiment_score)
    if leverage < SENTIMENT_THRESHOLD:
        return None, None, 0
    
    # Scale leverage (1-100)
    leverage = max(1, min(MAX_LEVERAGE, round(leverage)))
    
    # Calculate position size
    position_size = BASE_POSITION_SIZE
    
    return side, position_size, leverage

def execute_trade(side, position_size, leverage):
    """
    Execute a trade on Binance based on sentiment analysis
    """
    if side is None or position_size == 0:
        logger.info("No trade signal generated")
        return
    
    try:
        # Set leverage
        binance_client.futures_change_leverage(symbol=SYMBOL, leverage=leverage)
        logger.info(f"Set leverage to {leverage}x")
        
        # Close any existing positions
        try:
            binance_client.futures_create_order(
                symbol=SYMBOL,
                type="MARKET",
                side="BUY" if side == "SELL" else "SELL",
                reduceOnly=True,
                quantity=position_size
            )
            logger.info("Closed existing position")
        except BinanceAPIException as e:
            if "reduceOnly" not in str(e):
                logger.error(f"Error closing position: {str(e)}")
        
        # Open new position
        order = binance_client.futures_create_order(
            symbol=SYMBOL,
            side=side,
            type="MARKET",
            quantity=position_size
        )
        
        logger.info(f"Executed {side} order for {position_size} {SYMBOL} with {leverage}x leverage")
        logger.info(f"Order details: {order}")
        
    except BinanceAPIException as e:
        logger.error(f"Binance API error: {str(e)}")
    except Exception as e:
        logger.error(f"Error executing trade: {str(e)}")

async def trading_loop():
    """
    Main trading loop
    """
    logger.info("Starting Bitcoin sentiment trading bot")
    
    while True:
        try:
            # 1. Fetch tweets
            tweets = await get_twitter_data()
            
            # 2. Analyze sentiment
            sentiment_score = await analyze_sentiment(tweets)
            
            # 3. Calculate position
            side, position_size, leverage = calculate_position_size_and_side(sentiment_score)
            
            # 4. Execute trade if signal is strong enough
            if leverage >= SENTIMENT_THRESHOLD:
                execute_trade(side, position_size, leverage)
            else:
                logger.info(f"Sentiment score ({sentiment_score}) below threshold, no trade executed")
            
            # Wait before next iteration
            logger.info("Waiting for next cycle...")
            await asyncio.sleep(3600)  # Run every hour
            
        except Exception as e:
            logger.error(f"Error in trading loop: {str(e)}")
            await asyncio.sleep(300)  # Wait 5 minutes on error

if __name__ == "__main__":
    try:
        asyncio.run(trading_loop())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}") 