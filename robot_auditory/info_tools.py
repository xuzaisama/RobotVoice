"""为聊天问答提供时间与天气查询工具。"""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from . import config


@dataclass(frozen=True)
class LocationInfo:
    name: str
    latitude: float
    longitude: float
    timezone: str


WEATHER_CODE_TEXT = {
    0: "晴",
    1: "大部晴朗",
    2: "多云",
    3: "阴",
    45: "有雾",
    48: "有雾并伴有霜",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "较强毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "小阵雨",
    81: "中等阵雨",
    82: "强阵雨",
    95: "雷暴",
    96: "伴有小冰雹的雷暴",
    99: "伴有大冰雹的雷暴",
}


def _make_ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _http_get_json(url: str, params: dict[str, object]) -> dict:
    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{query}", method="GET")
    with urllib.request.urlopen(req, timeout=20, context=_make_ssl_context()) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def should_answer_with_live_info(text: str) -> bool:
    text = text or ""
    weather_keywords = ("天气", "气温", "温度", "下雨", "下雪", "冷不冷", "热不热")
    time_keywords = ("几点", "时间", "几号", "日期", "星期", "周几", "现在几点")
    return any(k in text for k in weather_keywords + time_keywords)


def wants_weather(text: str) -> bool:
    text = text or ""
    return any(k in text for k in ("天气", "气温", "温度", "下雨", "下雪", "冷不冷", "热不热"))


def wants_time(text: str) -> bool:
    text = text or ""
    return any(k in text for k in ("几点", "时间", "几号", "日期", "星期", "周几", "现在几点"))


def extract_city_name(text: str) -> Optional[str]:
    text = (text or "").strip()
    for marker in ("天气", "气温", "温度"):
        if marker in text:
            prefix = text.split(marker, 1)[0].strip("，。！？? ")
            for noise in ("请告诉我", "请问", "告诉我", "查询", "看看", "现在", "当地", "今天", "一下", "请"):
                prefix = prefix.replace(noise, "")
            prefix = prefix.strip()
            if 1 < len(prefix) <= 12:
                return prefix
    return None


def resolve_location(
    city: Optional[str] = None,
    fetch_json: Callable[[str, dict[str, object]], dict] = _http_get_json,
) -> LocationInfo:
    target_city = city or config.LOCAL_CITY
    data = fetch_json(
        config.WEATHER_GEOCODING_URL,
        {
            "name": target_city,
            "count": 1,
            "language": "zh",
            "format": "json",
        },
    )
    results = data.get("results") or []
    if not results:
        raise RuntimeError(f"未找到城市：{target_city}")

    item = results[0]
    return LocationInfo(
        name=str(item.get("name") or target_city),
        latitude=float(item["latitude"]),
        longitude=float(item["longitude"]),
        timezone=str(item.get("timezone") or config.LOCAL_TIMEZONE),
    )


def get_time_summary(location: Optional[LocationInfo] = None) -> str:
    tz_name = location.timezone if location else config.LOCAL_TIMEZONE
    city_name = location.name if location else config.LOCAL_CITY
    now = datetime.now(ZoneInfo(tz_name))
    weekday_map = "一二三四五六日"
    weekday_text = weekday_map[now.weekday()]
    return (
        f"{city_name}当前时间是{now.year}年{now.month:02d}月{now.day:02d}日，"
        f"星期{weekday_text}，{now.hour:02d}:{now.minute:02d}:{now.second:02d}。"
    )


def get_weather_summary(
    location: LocationInfo,
    fetch_json: Callable[[str, dict[str, object]], dict] = _http_get_json,
) -> str:
    data = fetch_json(
        config.WEATHER_API_URL,
        {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
            "timezone": location.timezone,
        },
    )
    current = data.get("current") or {}
    temp = current.get("temperature_2m")
    apparent = current.get("apparent_temperature")
    humidity = current.get("relative_humidity_2m")
    weather_code = int(current.get("weather_code", -1))
    wind_speed = current.get("wind_speed_10m")
    weather_text = WEATHER_CODE_TEXT.get(weather_code, "天气未知")
    return (
        f"{location.name}当前天气{weather_text}，气温{temp}摄氏度，"
        f"体感温度{apparent}摄氏度，相对湿度{humidity}%，风速{wind_speed}公里每小时。"
    )


def build_live_info_context(
    text: str,
    fetch_json: Callable[[str, dict[str, object]], dict] = _http_get_json,
) -> Optional[str]:
    if not should_answer_with_live_info(text):
        return None

    city = extract_city_name(text)
    location = None
    lines: list[str] = []

    if wants_weather(text) or city:
        location = resolve_location(city=city, fetch_json=fetch_json)
        lines.append(get_weather_summary(location, fetch_json=fetch_json))

    if wants_time(text):
        if location is None and city:
            location = resolve_location(city=city, fetch_json=fetch_json)
        lines.append(get_time_summary(location))

    if not lines:
        return None

    return "以下是工具实时查询结果，请基于这些结果直接回答用户，不要编造天气或时间。\n" + "\n".join(lines)
