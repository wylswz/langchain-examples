import re
def remove_think(text: str) -> str:
    """
    Removes <think> tags and their contents from a document.
    
    Args:
        text (str): Input text containing <think> tags
        
    Returns:
        str: Text with all <think> tags removed
    """
    # Pattern matches <think> followed by any characters (including newlines) until </think>
    pattern = r'<think>.*?</think>'
    # Use DOTALL flag to match across multiple lines
    return re.sub(pattern, '', text, flags=re.DOTALL).strip()


if __name__ == '__main__':
    print(remove_think("""
aaa
<think>
                       
                       asdasdzxc
                       zxc
                       zxc
                       zxc
                       </think>
"""))