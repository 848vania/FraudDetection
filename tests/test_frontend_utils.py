from frontend.utils import format_cost, format_percent, format_latency_ms


def test_format_percent():
    assert format_percent(0.1234) == '12.3%'
    assert format_percent(None) == '-'


def test_format_cost():
    assert format_cost(1234.5) == "1234.50"
    assert  format_cost(None) == '-'


def test_format_latency_ms():
    assert format_latency_ms(23.456) == '23.5 ms'
    assert format_latency_ms(None) == '-'