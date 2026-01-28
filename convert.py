import json

def build_trie(words):
    trie = {}
    for word in words:
        current = trie
        for char in word:
            if char not in current:
                current[char] = {}
            current = current[char]
        current['isWord'] = True
    
    return trie

def load_and_convert(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    words = data['commonWords']
    trie = build_trie(words)
    with open(output_file, 'w') as f:
        json.dump(trie, f, indent=2)
    print(f"Trie structure created with {len(words)} words")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    load_and_convert('common.json', 'common_trie.json')