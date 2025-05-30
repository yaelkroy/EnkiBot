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
import logging
import httpx 
from datetime import datetime
from typing import Dict, Any, Optional

from enkibot.core.llm_services import LLMServices 

logger = logging.getLogger(__name__)

class ApiRouter:
    def __init__(self, weather_api_key: str | None, news_api_key: str | None, llm_services: LLMServices):
        self.weather_api_key = weather_api_key
        self.news_api_key = news_api_key
        self.llm_services = llm_services
        self.lang_to_country_map = {
            "en": "us", "ru": "ru", "de": "de", "fr": "fr", "es": "es", 
            "it": "it", "ja": "jp", "ko": "kr", "zh": "cn", "bg": "bg",
            "ua": "ua", "pl": "pl", "tr": "tr", "pt": "pt", # "br" might be better for pt for news
        }
        self.default_news_country = "us"

    def _get_localized_response(self, response_strings: Dict[str, str], key: str, default_value: str, **kwargs) -> str:
        raw_string = response_strings.get(key, default_value)
        try:
            return raw_string.format(**kwargs) if kwargs else raw_string
        except KeyError as e:
            logger.error(f"Missing format key '{e}' in response string for key '{key}'. Raw: '{raw_string}'")
            return default_value # Or a more generic error placeholder

    async def get_weather_data(self, location: str, forecast_type: str = 'current', days: int = 7, 
                               lang_pack_full: Optional[Dict[str, Any]] = None) -> str:
        
        response_strings = lang_pack_full.get("responses", {}) if lang_pack_full else {}
        weather_conditions_map = lang_pack_full.get("weather_conditions_map", {}) if lang_pack_full else {}
        days_of_week_map = lang_pack_full.get("days_of_week", {}) if lang_pack_full else {}

        if not self.weather_api_key:
            logger.warning("Weather API key is not configured.")
            return self._get_localized_response(response_strings, "weather_api_key_missing", "Weather service: API key missing.")

        api_lang = self._get_localized_response(response_strings, "api_lang_code_openweathermap", "en")

        if forecast_type == 'current':
            url, params = "https://api.openweathermap.org/data/2.5/weather", {"q": location, "appid": self.weather_api_key, "units": "metric", "lang": api_lang}
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, params=params); response.raise_for_status()
                data = response.json()
                city = data.get("name", location)
                desc_key = data["weather"][0].get("description", "unknown_condition").lower().replace(" ", "_")
                desc = weather_conditions_map.get(desc_key, data["weather"][0].get("description", "N/A").capitalize())
                
                return "\n".join([
                    self._get_localized_response(response_strings, "weather_report_intro_current", "Current weather in {city}:", city=city),
                    self._get_localized_response(response_strings, "weather_current_conditions", "  - Currently: {description}", description=desc),
                    self._get_localized_response(response_strings, "weather_temperature", "  - Temperature: {temp:.1f}°C (feels like {feels_like:.1f}°C)", temp=data["main"]["temp"], feels_like=data["main"]["feels_like"]),
                    self._get_localized_response(response_strings, "weather_wind", "  - Wind: {wind_speed:.1f} m/s", wind_speed=data["wind"]["speed"])
                ])
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404: return self._get_localized_response(response_strings, "weather_city_not_found", "Sorry, I couldn't find the city '{location}'.", location=location)
                logger.error(f"HTTP error current weather {location}: {e}")
                return self._get_localized_response(response_strings, "weather_server_error", "Could not get weather data due to a server error.")
            except Exception as e:
                logger.error(f"Unexpected error current weather: {e}", exc_info=True)
                return self._get_localized_response(response_strings, "weather_unexpected_error", "An unexpected error occurred while fetching weather.")

        elif forecast_type == 'forecast':
            url, params = "https://api.openweathermap.org/data/2.5/forecast", {"q": location, "appid": self.weather_api_key, "units": "metric", "lang": api_lang, "cnt": 40}
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, params=params); response.raise_for_status()
                data = response.json()
                city = data.get("city", {}).get("name", location)
                if not data.get("list"): return self._get_localized_response(response_strings, "weather_forecast_unavailable", "Forecast data unavailable for '{location}'.", location=location)

                daily_summary = {}
                for item in data["list"]:
                    day_str = datetime.fromtimestamp(item["dt"]).strftime('%Y-%m-%d')
                    if day_str not in daily_summary:
                        eng_day_name = datetime.fromtimestamp(item["dt"]).strftime('%A')
                        daily_summary[day_str] = {'day_name': days_of_week_map.get(eng_day_name, eng_day_name), 'temps': [], 'descs': set()}
                    daily_summary[day_str]['temps'].append(item['main']['temp'])
                    desc_key = item['weather'][0].get("description", "unknown_condition").lower().replace(" ", "_")
                    daily_summary[day_str]['descs'].add(weather_conditions_map.get(desc_key, item['weather'][0].get("description", "").capitalize()))
                
                report = [self._get_localized_response(response_strings, "weather_report_intro_forecast", "Weather forecast for {city}:", city=city)]
                processed_days = 0
                for day_str_sorted in sorted(daily_summary.keys()):
                    if processed_days >= days: break
                    day_data = daily_summary[day_str_sorted]
                    avg_temp = sum(day_data['temps']) / len(day_data['temps'])
                    # Prioritize common descriptions or just join them
                    desc_text = ", ".join(filter(None, day_data['descs'])) or "N/A"
                    report.append(self._get_localized_response(response_strings, "weather_forecast_day_item", "  - {day_name}: {temp:.0f}°C, {description}", day_name=day_data['day_name'], temp=avg_temp, description=desc_text))
                    processed_days += 1
                return "\n".join(report)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404: return self._get_localized_response(response_strings, "weather_city_not_found_forecast", "Sorry, I couldn't find '{location}' for the forecast.", location=location)
                logger.error(f"HTTP error forecast {location}: {e}")
                return self._get_localized_response(response_strings, "weather_server_error_forecast", "Could not get forecast data: server error.")
            except Exception as e:
                logger.error(f"Unexpected error forecast: {e}", exc_info=True)
                return self._get_localized_response(response_strings, "weather_unexpected_error_forecast", "Unexpected error fetching forecast.")
        return self._get_localized_response(response_strings, "weather_unknown_type", "Unknown weather request type.")

    async def get_latest_news(self, query: str | None = None, lang_code: str = "en", 
                              num: int = 5, response_strings: Optional[Dict[str, str]] = None) -> str:
        if not response_strings: response_strings = {} # Ensure it's a dict
        if not self.news_api_key:
            logger.warning("News API key is not configured.")
            return self._get_localized_response(response_strings, "news_api_key_missing", "News service: API key missing.")

        params: Dict[str, Any] = {"apiKey": self.news_api_key, "pageSize": num}
        base_url, endpoint = "https://newsapi.org/v2/", ""

        if query:
            logger.info(f"Fetching news for query: '{query}', language: {lang_code}")
            endpoint, params["q"], params["language"], params["sortBy"] = "everything", query, lang_code, "relevancy"
        else:
            country = self.lang_to_country_map.get(lang_code, self.default_news_country)
            logger.info(f"Fetching top headlines for country: '{country}' (from lang: {lang_code})")
            endpoint, params["country"], params["category"] = "top-headlines", country, "general"
        
        url = base_url + endpoint
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, params=params); logger.debug(f"NewsAPI URL: {resp.url}"); resp.raise_for_status()
            data, articles = resp.json(), resp.json().get("articles", [])
            logger.info(f"NewsAPI {len(articles)} articles (total: {data.get('totalResults')}) for {params}")

            if not articles:
                return self._get_localized_response(response_strings, "news_api_no_articles" if query else "news_api_no_general_articles", 
                                                 "No news found for '{query}'." if query else "No general news found.", query=query)

            title = self._get_localized_response(response_strings, "news_report_title_topic" if query else "news_report_title_general",
                                              "News on '{topic}':" if query else "Latest News:", topic=query)
            headlines = [title] + [
                f"- {a.get('title','N/A')} ({a.get('source',{}).get('name','N/A')})\n  {self._get_localized_response(response_strings, 'news_read_more', 'Read: {url}', url=a.get('url','#'))}" 
                for a in articles
            ]
            return "\n\n".join(headlines)
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error news ({url}): {e.response.status_code} - {e.response.text}")
            return self._get_localized_response(response_strings, "news_api_error", "Could not fetch news. Service error {status_code}.", status_code=e.response.status_code)
        except Exception as e:
            logger.error(f"Unexpected error news: {e}", exc_info=True)
            return self._get_localized_response(response_strings, "news_unexpected_error", "Unexpected error fetching news.")