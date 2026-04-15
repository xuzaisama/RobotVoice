"""时间/天气工具层测试。"""

import unittest

from robot_auditory.info_tools import (
    build_live_info_context,
    extract_city_name,
    get_time_summary,
    should_answer_with_live_info,
)


def fake_fetch_json(url: str, params: dict[str, object]) -> dict:
    if "geocoding-api" in url:
        return {
            "results": [
                {
                    "name": "上海",
                    "latitude": 31.2222,
                    "longitude": 121.4581,
                    "timezone": "Asia/Shanghai",
                }
            ]
        }
    return {
        "current": {
            "temperature_2m": 24.5,
            "apparent_temperature": 25.1,
            "relative_humidity_2m": 68,
            "weather_code": 1,
            "wind_speed_10m": 9.2,
        }
    }


class TestInfoTools(unittest.TestCase):
    def test_detect_live_info_question(self) -> None:
        self.assertTrue(should_answer_with_live_info("现在几点"))
        self.assertTrue(should_answer_with_live_info("今天天气怎么样"))
        self.assertFalse(should_answer_with_live_info("讲个笑话"))

    def test_extract_city_name(self) -> None:
        self.assertEqual(extract_city_name("上海天气怎么样"), "上海")
        self.assertEqual(extract_city_name("请告诉我北京气温"), "北京")

    def test_build_live_info_context(self) -> None:
        context = build_live_info_context("上海今天天气怎么样", fetch_json=fake_fetch_json)
        self.assertIsNotNone(context)
        assert context is not None
        self.assertIn("上海当前天气", context)
        self.assertIn("24.5", context)

    def test_get_time_summary(self) -> None:
        summary = get_time_summary()
        self.assertIn("当前时间", summary)
        self.assertIn("星期", summary)


if __name__ == "__main__":
    unittest.main()
