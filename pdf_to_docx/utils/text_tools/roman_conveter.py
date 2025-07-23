ROMAN_MAP = {
    "I": 1, "V": 5, "X": 10, "L": 50,
    "C": 100, "D": 500, "M": 1000
}

def romanToInt(s: str) -> int:
    value = 0
    n = len(s)
    
    for i in range(n - 1):
        curr_val = ROMAN_MAP[s[i]]
        next_val = ROMAN_MAP[s[i + 1]]
        
        if curr_val < next_val:
            value -= curr_val
        else:
            value += curr_val
    
    value += ROMAN_MAP[s[n-1]]
    return value