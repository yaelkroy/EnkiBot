# enkibot/modules/api_router.py
# EnkiBot: Advanced Multilingual Telegram AI Assistant
# Copyright (C) 2025 Yael Demedetskaya <yaelkroy@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# ==================================================================================================
# === EnkiBot API Router ===
# ==================================================================================================

# (GPLv3 Header as in your files)
import logging
import httpx 
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from enkibot import config # For API Keys
# Removed LLMServices import as it's not directly used here anymore for extraction

logger = logging.getLogger(__name__)

class ApiRouter:
    def __init__(self, weather_api_key: str | None, news_api_key: str | None, llm_services: Any = None): # llm_services no longer strictly needed here
        self.weather_api_key = weather_api_key
        self.news_api_key = news_api_key
        # self.llm_services = llm_services # Not used directly in this version of ApiRouter

        self.lang_to_country_map = {
            "en": "us", "ru": "ru", "de": "de", "fr": "fr", "es": "es", 
            "it": "it", "ja": "jp", "ko": "kr", "zh": "cn", "bg": "bg",
            "ua": "ua", "pl": "pl", "tr": "tr", "pt": "pt", 
        }
        self.default_news_country = "us"

    def _get_localized_response_string_from_pack(self, lang_pack_full: Optional[Dict[str, Any]], key: str, default_value: str, **kwargs) -> str:
        """ Helper to get response strings directly from a full language pack. """
        if lang_pack_full and "responses" in lang_pack_full:
            raw_string = lang_pack_full["responses"].get(key, default_value)
        else: # Fallback if pack or responses section is missing
            raw_string = default_value
        try:
            return raw_string.format(**kwargs) if kwargs else raw_string
        except KeyError: return default_value


    async def get_weather_data_structured(self, location: str, lang_pack_full: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Fetches 5-day weather forecast data and returns it in a structured format
        suitable for LLM processing. Includes city name and a list of daily forecasts.
        """
        if not self.weather_api_key:
            logger.warning("Weather API key is not configured.")
            return None

        api_lang = "en" # Default to English for weather data city name, LLM can localize description
        if lang_pack_full and "responses" in lang_pack_full:
            api_lang = lang_pack_full["responses"].get("api_lang_code_openweathermap", "en")

        # OpenWeatherMap 5 day / 3 hour forecast endpoint
        url = "https://api.openweathermap.org/data/2.5/forecast"
        # Requesting enough data points for 5 days (8 records per day * 5 days = 40)
        params = {"q": location, "appid": self.weather_api_key, "units": "metric", "lang": api_lang, "cnt": 40}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
            data = response.json()

            city_name = data.get("city", {}).get("name", location)
            forecast_items = data.get("list", [])

            if not forecast_items:
                logger.warning(f"No forecast items received for {location}")
                return None

            daily_forecasts_processed = []
            temp_daily_data = {} # date_str -> {'temps': [], 'conditions': set(), 'icons': set()}

            for item in forecast_items:
                dt_object = datetime.fromtimestamp(item["dt"])
                date_str = dt_object.strftime("%Y-%m-%d")
                
                if date_str not in temp_daily_data:
                    temp_daily_data[date_str] = {'temps': [], 'conditions': set(), 'icons': set()}
                
                temp_daily_data[date_str]['temps'].append(item["main"]["temp"])
                if item.get("weather") and item["weather"][0]:
                    temp_daily_data[date_str]['conditions'].add(item["weather"][0].get("description", "N/A"))
                    temp_daily_data[date_str]['icons'].add(item["weather"][0].get("icon", "N/A"))

            for date_str, daily_data in sorted(temp_daily_data.items()):
                if not daily_data['temps']: continue
                
                # For simplicity, take the condition that appears most or first unique for the day.
                # LLM can make sense of multiple conditions if provided as a list.
                day_condition = list(daily_data['conditions'])[0] if daily_data['conditions'] else "N/A"
                
                daily_forecasts_processed.append({
                    "date": date_str,
                    "day_name": datetime.strptime(date_str, "%Y-%m-%d").strftime('%A'), # English day name
                    "temp_min": min(daily_data['temps']),
                    "temp_max": max(daily_data['temps']),
                    "avg_temp": sum(daily_data['temps']) / len(daily_data['temps']),
                    "condition_descriptions": list(daily_data['conditions']), # Provide all conditions
                    "primary_condition": day_condition # LLM can choose or summarize
                })
            
            if not daily_forecasts_processed:
                 logger.warning(f"Could not process daily forecast data for {location}")
                 return None

            return {
                "location": city_name,
                "forecast_days": daily_forecasts_processed[:7] # Return up to 7 days of processed data
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching weather forecast for {location}: {e.response.status_code} - {e.response.text[:200]}")
        except Exception as e:
            logger.error(f"Unexpected error fetching structured weather data for {location}: {e}", exc_info=True)
        return None

    async def get_latest_news_structured(self, query: Optional[str] = None, lang_code: str = "en", num: int = 5) -> Optional[List[Dict[str, Any]]]:
        """
        Fetches latest news and returns a list of article dictionaries.
        """
        if not self.news_api_key:
            logger.warning("News API key is not configured.")
            return None

        params: Dict[str, Any] = {"apiKey": self.news_api_key, "pageSize": num}
        base_url = "https://newsapi.org/v2/"
        endpoint: str

        if query:
            logger.info(f"Fetching news for query: '{query}', language: {lang_code}")
            endpoint = "everything"
            params.update({"q": query, "language": lang_code, "sortBy": "relevancy"})
        else:
            country = self.lang_to_country_map.get(lang_code, self.default_news_country)
            logger.info(f"Fetching top headlines for country: '{country}' (derived from lang: {lang_code})")
            endpoint = "top-headlines"
            params.update({"country": country, "category": "general"})
        
        url = base_url + endpoint
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, params=params)
                logger.debug(f"NewsAPI request URL: {resp.url}")
                resp.raise_for_status()
            data = resp.json()
            articles_raw = data.get("articles", [])
            
            logger.info(f"NewsAPI returned {len(articles_raw)} articles (totalResults: {data.get('totalResults')}) for params: {params}")

            if not articles_raw:
                return [] # Return empty list if no articles

            # Process articles into a cleaner structure for the LLM
            processed_articles = []
            for article in articles_raw:
                processed_articles.append({
                    "title": article.get("title"),
                    "source": article.get("source", {}).get("name"),
                    "description": article.get("description"), # Short description or snippet
                    "url": article.get("url"),
                    "published_at": article.get("publishedAt")
                })
            return processed_articles
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching news ({url}): {e.response.status_code} - {e.response.text[:200]}")
        except Exception as e:
            logger.error(f"Unexpected error fetching structured news: {e}", exc_info=True)
        return None