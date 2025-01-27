import pytest
from utilities import extract_options

@pytest.mark.parametrize("test_input,expected", [
    # Test case 1: JSON-like list with LaTeX math
    (
        '["`144π cm^3`", "`162π cm^3`", "`216π cm^3`", "`432π cm^3`", "`288π cm^3`"]',
        ['`144π cm^3`', '`162π cm^3`', '`216π cm^3`', '`432π cm^3`', '`288π cm^3`']
    ),
    
    # Test case 2: Simple YAML list with numbers as strings
    (
        "---\n- '15'\n- '20'\n- '30'\n- '35'\n- '25'\n",
        ['15', '20', '30', '35', '25']
    ),
    
    # Test case 3: YAML list with complex LaTeX expressions
    (
        "---\n- \"`471` `\\\\text{cm}^2`\"\n- \"`706,5` `\\\\text{cm}^2`\"\n- \"`117,5` `\\\\text{cm}^2`\"\n- \"`31,4` `\\\\text{cm}^2`\"\n- \"`15,7` `\\\\text{cm}^2`\"\n",
        ['`471` `\\text{cm}^2`', '`706,5` `\\text{cm}^2`', '`117,5` `\\text{cm}^2`', '`31,4` `\\text{cm}^2`', '`15,7` `\\text{cm}^2`']
    ),
    
    # Test case 4: YAML list with fractions
    (
        "---\n- \"`10`\"\n- \"`1/10`\"\n- \"`7/10`\"\n- \"`-7/10`\"\n- \"`-1/10`\"\n",
        ['`10`', '`1/10`', '`7/10`', '`-7/10`', '`-1/10`']
    ),
    
    # Test case 5: YAML list with text containing equals signs
    (
        "---\n- a= 2 y b = 20\n- a= 4 y b =10\n- a= 2 y b = 1\n- a = 1 y b = 120\n- a= 2 y b = 30\n",
        ['a= 2 y b = 20', 'a= 4 y b =10', 'a= 2 y b = 1', 'a = 1 y b = 120', 'a= 2 y b = 30']
    ),
    
    # Test case 6: YAML list with pi and special characters
    (
        "---\n- \"`4/3\\\\pi` cm\"\n- \"`1/3\\\\pi` cm\"\n- \"`2/3\\\\pi` cm\"\n- \"`2\\\\pi` cm\"\n- \"`4\\\\pi` cm\"\n",
        ['`4/3\\pi` cm', '`1/3\\pi` cm', '`2/3\\pi` cm', '`2\\pi` cm', '`4\\pi` cm']
    ),
    
    # Test case 7: YAML list with decimal numbers
    (
        "---\n- \"`3,5`\"\n- \"`4`\"\n- \"`2,75`\"\n- \"`2,5`\"\n- \"`3`\"\n",
        ['`3,5`', '`4`', '`2,75`', '`2,5`', '`3`']
    ),
    
    # Test case 8: YAML list with algebraic expressions
    (
        "---\n- \"`x+2y`\"\n- \"`5x+10y`\"\n- \"`5x-2y`\"\n- \"`3x-y`\"\n- \"`5x+2y`\"\n",
        ['`x+2y`', '`5x+10y`', '`5x-2y`', '`3x-y`', '`5x+2y`']
    ),
    
    # Test case 9: Edge case - malformed input
    (
        "invalid[format]",
        [None, None, None, None, None]
    ),
    
    # Test case 10: Edge case - empty string
    (
        "",
        [None, None, None, None, None]
    ),
    
    # Test case 11: Edge case - None input
    (
        None,
        [None, None, None, None, None]
    ),
    
    # Test case 12: YAML list with ratios
    (
        "---\n- \"`4` : `8`\"\n- \"`2` : `9`\"\n- \"`sqrt(2): sqrt(9)`\"\n- \"`4` : `81`\"\n- \"`1` : `81`\"\n",
        ['`4` : `8`', '`2` : `9`', '`sqrt(2): sqrt(9)`', '`4` : `81`', '`1` : `81`']
    )
])
def test_extract_options(test_input, expected):
    result = extract_options(test_input)
    assert result == expected, f"Failed for input: {test_input}\nExpected: {expected}\nGot: {result}"

# Additional test for specific error cases
def test_extract_options_with_invalid_yaml():
    invalid_yaml = "---\n- 'unclosed string\n- 'another string'\n"
    result = extract_options(invalid_yaml)
    assert result == [None, None, None, None, None]

def test_extract_options_with_invalid_json():
    invalid_json = "[1, 2, 3, 'unclosed string]"
    result = extract_options(invalid_json)
    assert result == [None, None, None, None, None] 