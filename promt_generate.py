"""
Prompt Generator for Different Head Types

This module generates prompts that match the patterns required for detecting
different types of attention heads in transformer models.

Supported head types:
- duplicate_token_head: Prompts with duplicate token sequences
- induction_head: Prompts with repeated patterns (A B C A B C)
- previous_token_head: Normal text sequences
- truthfulness: Factual questions
- retrieval: Questions requiring information retrieval
- iteration: Iterative patterns
"""

import random
from typing import List, Dict, Tuple
from collections import defaultdict


def get_length_distribution(max_length: int = 100, min_length: int = None, 
                           interval: int = None, length_per_interval: int = None) -> Dict[int, int]:
    """
    Get the distribution of prompt lengths and number of prompts per length.
    
    Args:
        max_length: Maximum prompt length (default: 100)
        min_length: Minimum prompt length (default: 10, or uses default strategy)
        interval: Interval between lengths (default: None, uses default strategy)
        length_per_interval: Number of prompts per interval length (default: None, uses default strategy)
        
    Returns:
        Dictionary mapping length to number of prompts to generate
        
    Examples:
        # Default strategy (10-100 with varying counts)
        >>> dist = get_length_distribution(max_length=100)
        
        # Custom: every 10 tokens from 10 to 100, 5 prompts each
        >>> dist = get_length_distribution(max_length=100, min_length=10, interval=10, length_per_interval=5)
    """
    # If custom parameters are provided, use them
    if min_length is not None and interval is not None and length_per_interval is not None:
        distribution = {}
        # Generate lengths at intervals
        current_length = min_length
        while current_length <= max_length:
            distribution[current_length] = length_per_interval
            current_length += interval
        return distribution
    
    # Default distribution strategy (backward compatible)
    distribution = {}
    
    # 10-20: 5 prompts per length
    for length in range(10, 21):
        if length <= max_length:
            distribution[length] = 5
    
    # 21-40: 3 prompts per length
    for length in range(21, 41):
        if length <= max_length:
            distribution[length] = 3
    
    # 41-60: 2 prompts per length
    for length in range(41, 61):
        if length <= max_length:
            distribution[length] = 2
    
    # 61-80: 1 prompt per length
    for length in range(61, 81):
        if length <= max_length:
            distribution[length] = 1
    
    # 81-100: 1 prompt per length
    for length in range(81, max_length + 1):
        distribution[length] = 1
    
    return distribution


def generate_duplicate_token_prompt(target_length: int, seed: int = None) -> str:
    """Generate a prompt with duplicate token patterns for duplicate_token_head detection.
    
    Format examples:
    - "one two three one two three one two three"
    - "1 2 3 4 5 1 2 3 4 1 2 3 1 2 3 4 5 6 7"
    """
    if seed is not None:
        random.seed(seed)
    
    # Use simple patterns like the examples
    # Pattern 1: Word sequences (one two three one two three) - Expanded
    word_sequences = [
        ["one", "two", "three"],
        ["red", "blue", "green"],
        ["cat", "dog", "bird"],
        ["apple", "banana", "orange"],
        ["first", "second", "third"],
        ["up", "down", "left", "right"],
        ["spring", "summer", "fall", "winter"],
        ["monday", "tuesday", "wednesday"],
        ["january", "february", "march"],
        ["sun", "moon", "star"],
        ["water", "fire", "earth", "air"],
        ["north", "south", "east", "west"],
        ["happy", "sad", "angry"],
        ["big", "small", "medium"],
        ["fast", "slow", "quick"],
        ["hot", "cold", "warm"],
        ["new", "old", "young"],
        ["good", "bad", "nice"],
        ["high", "low", "middle"],
        ["day", "night", "morning", "evening"]
    ]
    
    # Pattern 2: Number sequences (1 2 3 4 5 1 2 3 4) - Expanded range
    # Numbers can now go up to 20 for longer sequences
    max_number = 20
    
    # Choose pattern type (50% word, 50% number)
    use_numbers = random.random() < 0.5
    
    tokens = []
    current_length = 0
    
    while current_length < target_length:
        remaining = target_length - current_length
        
        if use_numbers:
            # Use number sequences
            if remaining < 2:
                # Fill with single numbers (expanded range: 1-20)
                tokens.append(str(random.randint(1, max_number)))
                current_length += 1
                continue
            
            # Select a number sequence (2-10 numbers, up to max_number)
            max_seq_len = min(10, remaining // 2, max_number)  # Need space for repetition
            seq_len = max(2, min(max_seq_len, remaining // 2))
            # Start from random position to add variety
            start_num = random.randint(1, max(1, max_number - seq_len + 1))
            base_seq = list(range(start_num, start_num + seq_len))
            sequence = [str(x) for x in base_seq]
        else:
            # Use word sequences
            if remaining < 2:
                # Fill with single words
                all_words = [w for seq in word_sequences for w in seq]
                tokens.append(random.choice(all_words))
                current_length += 1
                continue
            
            # Select a word sequence (2-6 words, expanded)
            max_seq_len = min(6, remaining // 2)
            seq_len = max(2, min(max_seq_len, remaining // 2))
            sequence = random.choice(word_sequences)[:seq_len]
            # If sequence is shorter, extend it from all available words
            if len(sequence) < seq_len:
                all_words = [w for seq in word_sequences for w in seq]
                while len(sequence) < seq_len:
                    word = random.choice(all_words)
                    if word not in sequence:
                        sequence.append(word)
        
        # Add the sequence
        tokens.extend(sequence)
        current_length += len(sequence)
        
        # Repeat the sequence if we have space (matching the example pattern)
        if current_length < target_length and (target_length - current_length) >= len(sequence):
            tokens.extend(sequence)
            current_length += len(sequence)
        elif current_length < target_length:
            # Partial repeat if space allows
            partial_len = min(len(sequence), target_length - current_length)
            tokens.extend(sequence[:partial_len])
            current_length += partial_len
    
    # Trim to exact length
    return " ".join(tokens[:target_length])


def generate_induction_prompt(target_length: int, seed: int = None) -> str:
    """Generate a prompt with induction patterns (A B C A B C) for induction_head detection.
    
    Format examples (same as duplicate_token_head):
    - "one two three one two three one two three"
    - "1 2 3 4 5 1 2 3 4 1 2 3 1 2 3 4 5 6 7"
    """
    if seed is not None:
        random.seed(seed)
    
    # Use same patterns as duplicate_token_head (induction heads detect repeated patterns)
    # Pattern 1: Word sequences (one two three one two three) - Expanded
    word_sequences = [
        ["one", "two", "three"],
        ["red", "blue", "green"],
        ["cat", "dog", "bird"],
        ["apple", "banana", "orange"],
        ["first", "second", "third"],
        ["up", "down", "left", "right"],
        ["spring", "summer", "fall", "winter"],
        ["monday", "tuesday", "wednesday"],
        ["january", "february", "march"],
        ["sun", "moon", "star"],
        ["water", "fire", "earth", "air"],
        ["north", "south", "east", "west"],
        ["happy", "sad", "angry"],
        ["big", "small", "medium"],
        ["fast", "slow", "quick"],
        ["hot", "cold", "warm"],
        ["new", "old", "young"],
        ["good", "bad", "nice"],
        ["high", "low", "middle"],
        ["day", "night", "morning", "evening"]
    ]
    
    # Pattern 2: Number sequences - Expanded range
    max_number = 20
    
    # Choose pattern type (50% word, 50% number)
    use_numbers = random.random() < 0.5
    
    tokens = []
    current_length = 0
    
    while current_length < target_length:
        remaining = target_length - current_length
        
        if use_numbers:
            # Use number sequences
            if remaining < 2:
                tokens.append(str(random.randint(1, max_number)))
                current_length += 1
                continue
            
            # Select a number sequence (2-10 numbers, up to max_number)
            max_seq_len = min(10, remaining // 2, max_number)
            seq_len = max(2, min(max_seq_len, remaining // 2))
            # Start from random position to add variety
            start_num = random.randint(1, max(1, max_number - seq_len + 1))
            base_seq = list(range(start_num, start_num + seq_len))
            sequence = [str(x) for x in base_seq]
        else:
            # Use word sequences
            if remaining < 2:
                all_words = [w for seq in word_sequences for w in seq]
                tokens.append(random.choice(all_words))
                current_length += 1
                continue
            
            # Select a word sequence (2-6 words, expanded)
            max_seq_len = min(6, remaining // 2)
            seq_len = max(2, min(max_seq_len, remaining // 2))
            sequence = random.choice(word_sequences)[:seq_len]
            # Extend if needed
            if len(sequence) < seq_len:
                all_words = [w for seq in word_sequences for w in seq]
                while len(sequence) < seq_len:
                    word = random.choice(all_words)
                    if word not in sequence:
                        sequence.append(word)
        
        # Add the sequence
        tokens.extend(sequence)
        current_length += len(sequence)
        
        # Repeat the sequence (induction pattern: A B C A B C)
        if current_length < target_length and (target_length - current_length) >= len(sequence):
            tokens.extend(sequence)
            current_length += len(sequence)
        elif current_length < target_length:
            # Partial repeat if space allows
            partial_len = min(len(sequence), target_length - current_length)
            tokens.extend(sequence[:partial_len])
            current_length += partial_len
    
    return " ".join(tokens[:target_length])


def generate_previous_token_prompt(target_length: int, seed: int = None) -> str:
    """Generate a normal text prompt for previous_token_head detection.
    
    Format examples:
    - "The head detector feature for TransformerLens allows users to check for various common heads automatically, reducing the cost of discovery."
    - "Machine learning models require careful evaluation to ensure they perform well on unseen data."
    - "Attention mechanisms in transformers allow models to focus on relevant parts of the input sequence."
    """
    if seed is not None:
        random.seed(seed)
    
    # Base sentences matching the example style (technical/academic) - Expanded
    base_sentences = [
        "The head detector feature for TransformerLens allows users to check for various common heads automatically, reducing the cost of discovery.",
        "Machine learning models require careful evaluation to ensure they perform well on unseen data.",
        "Attention mechanisms in transformers allow models to focus on relevant parts of the input sequence.",
        "Neural networks process information through multiple layers of interconnected nodes.",
        "Deep learning architectures enable complex pattern recognition in large datasets.",
        "Transformer models use self-attention to capture relationships between tokens.",
        "Research in artificial intelligence continues to advance our understanding of cognitive processes.",
        "Computational methods provide powerful tools for analyzing complex systems.",
        "Data science combines statistical analysis with machine learning techniques.",
        "Natural language processing enables computers to understand and generate human language.",
        "Computer vision systems analyze visual information to extract meaningful patterns and features.",
        "Reinforcement learning algorithms learn optimal strategies through interaction with environments.",
        "Graph neural networks process structured data represented as graphs and networks.",
        "Generative models create new data samples that resemble training examples.",
        "Transfer learning techniques adapt pre-trained models to new tasks efficiently.",
        "Optimization algorithms find optimal solutions to complex mathematical problems.",
        "Feature engineering transforms raw data into meaningful representations for machine learning.",
        "Model interpretability helps understand how artificial intelligence systems make decisions.",
        "Distributed computing systems process large-scale data across multiple machines.",
        "Cloud computing infrastructure provides scalable resources for machine learning workloads.",
        "Edge computing brings computation closer to data sources for faster processing.",
        "Quantum computing explores new computational paradigms using quantum mechanical principles.",
        "Cybersecurity measures protect systems from malicious attacks and unauthorized access.",
        "Blockchain technology enables secure and transparent distributed ledger systems."
    ]
    
    # If target length is very short, use a short sentence - Expanded
    if target_length < 10:
        short_sentences = [
            "Machine learning models process data.",
            "Neural networks learn patterns.",
            "Transformers use attention mechanisms.",
            "Deep learning enables recognition.",
            "AI research advances understanding.",
            "Computer vision analyzes images.",
            "Natural language processing understands text.",
            "Reinforcement learning optimizes strategies.",
            "Data science extracts insights.",
            "Algorithms solve complex problems."
        ]
        sentence = random.choice(short_sentences)
        words = sentence.split()
        return " ".join(words[:target_length])
    
    # Start with a base sentence
    base_sentence = random.choice(base_sentences)
    words = base_sentence.split()
    
    # If base sentence is shorter than target, extend it
    if len(words) < target_length:
        # Additional phrases to extend the sentence - Expanded
        extensions = [
            "This approach", "provides", "significant", "improvements", "in", "performance", "and", "efficiency.",
            "These", "techniques", "enable", "researchers", "to", "analyze", "complex", "patterns", "more", "effectively.",
            "The", "methodology", "combines", "multiple", "approaches", "to", "achieve", "better", "results.",
            "Recent", "advances", "have", "demonstrated", "the", "potential", "of", "these", "technologies.",
            "Understanding", "these", "mechanisms", "is", "crucial", "for", "developing", "more", "robust", "systems.",
            "Experimental", "results", "show", "that", "this", "method", "outperforms", "previous", "approaches", "significantly.",
            "The", "framework", "supports", "various", "applications", "including", "classification", "regression", "and", "clustering.",
            "Implementation", "details", "include", "optimization", "strategies", "for", "computational", "efficiency", "and", "accuracy.",
            "Future", "work", "will", "explore", "extensions", "to", "handle", "more", "complex", "scenarios.",
            "The", "system", "demonstrates", "scalability", "across", "different", "data", "sizes", "and", "domains.",
            "Performance", "metrics", "indicate", "substantial", "gains", "in", "both", "speed", "and", "quality.",
            "Comparative", "analysis", "reveals", "advantages", "over", "traditional", "methods", "in", "various", "settings."
        ]
        
        # Add words from extensions until we reach target length
        needed = target_length - len(words)
        extension_words = " ".join(extensions).split()
        words.extend(extension_words[:needed])
    elif len(words) > target_length:
        # Trim to target length
        words = words[:target_length]
    
    return " ".join(words)


def generate_truthfulness_prompt(target_length: int, seed: int = None) -> str:
    """Generate a factual question prompt for truthfulness head detection."""
    if seed is not None:
        random.seed(seed)
    
    question_starters = [
        "What is", "Who is", "When did", "Where is", "Why does", "How does",
        "Is it true that", "Can you explain", "Tell me about",
        "What are the", "Which of the", "Are there",
        "What was", "Who was", "When was", "Where was", "Why was", "How was",
        "What were", "Who were", "When were", "Where were", "Why were", "How were",
        "What does", "Who does", "When does", "Where does", "Why do", "How do",
        "What did", "Who did", "When did", "Where did", "Why did", "How did",
        "Can you tell me", "Do you know", "Is it correct that", "Would you say that",
        "What do you know about", "What can you tell me about", "Explain to me"
    ]
    
    factual_topics = [
        "the capital of France", "the largest planet in our solar system",
        "the speed of light", "the chemical formula for water",
        "the year World War II ended", "the inventor of the telephone",
        "the number of continents", "the deepest ocean",
        "the first man on the moon", "the longest river in the world",
        "the smallest country", "the highest mountain",
        "the human body temperature", "the boiling point of water",
        "the distance from Earth to the Sun", "the number of bones in human body",
        "the capital of Japan", "the largest ocean", "the smallest planet",
        "the inventor of the light bulb", "the year the internet was created",
        "the number of planets in our solar system", "the tallest building in the world",
        "the fastest land animal", "the largest mammal", "the smallest mammal",
        "the freezing point of water", "the atomic number of carbon",
        "the number of chromosomes in humans", "the speed of sound",
        "the distance from Earth to the Moon", "the number of days in a year",
        "the largest desert", "the longest bridge", "the deepest lake",
        "the oldest civilization", "the largest country by area",
        "the most spoken language", "the number of elements in the periodic table",
        "the inventor of the computer", "the year the first computer was built",
        "the largest volcano", "the longest coastline", "the highest waterfall"
    ]
    
    # Generate question
    starter = random.choice(question_starters)
    topic = random.choice(factual_topics)
    
    prompt = f"{starter} {topic}"
    
    # Pad or trim to target length
    words = prompt.split()
    if len(words) < target_length:
        # Add more context - Expanded
        additional = [
            "Please provide", "accurate information", "about", "this", "topic",
            "based on", "scientific", "facts", "and", "evidence",
            "Give me", "detailed", "information", "regarding", "this", "subject",
            "I need", "to know", "more", "about", "this", "concept",
            "Can you", "provide", "specific", "details", "on", "this", "matter",
            "Tell me", "everything", "you", "know", "about", "this", "topic",
            "What", "are", "the", "key", "facts", "concerning", "this", "issue",
            "I would", "like", "to", "learn", "more", "about", "this", "subject"
        ]
        needed = target_length - len(words)
        # Use random.choices with replacement if needed > available
        if needed <= len(additional):
            words.extend(random.sample(additional, needed))
        else:
            # Sample with replacement if we need more words than available
            words.extend(random.choices(additional, k=needed))
    elif len(words) > target_length:
        words = words[:target_length]
    
    return " ".join(words)


def generate_retrieval_prompt(target_length: int, seed: int = None) -> str:
    """Generate a prompt requiring information retrieval for retrieval head detection."""
    if seed is not None:
        random.seed(seed)
    
    retrieval_patterns = [
        "Find information about", "Search for details on", "Retrieve data regarding",
        "Look up", "What do we know about", "Can you find",
        "Please retrieve", "I need information on", "Tell me what you know about",
        "Get me information about", "Search for", "Find details on",
        "Retrieve information regarding", "Look for information about",
        "What information is available about", "Can you search for",
        "I'm looking for information about", "Please find information on",
        "What can you tell me about", "Do you have information on",
        "I need to find", "Search the database for", "Query information about",
        "Extract information regarding", "Gather data about", "Collect information on"
    ]
    
    retrieval_topics = [
        "quantum computing", "artificial intelligence", "climate change",
        "renewable energy", "space exploration", "genetic engineering",
        "neural networks", "machine learning", "deep learning",
        "natural language processing", "computer vision", "robotics",
        "blockchain technology", "cryptocurrency", "cybersecurity",
        "biotechnology", "nanotechnology", "renewable resources",
        "quantum mechanics", "particle physics", "astrophysics",
        "molecular biology", "genetics", "evolution",
        "climate science", "environmental science", "ecology",
        "renewable energy sources", "solar power", "wind energy",
        "nuclear energy", "fossil fuels", "energy storage",
        "space technology", "satellite systems", "space missions",
        "medical research", "pharmaceuticals", "drug development",
        "data science", "big data", "data analytics",
        "cloud computing", "distributed systems", "parallel computing",
        "internet of things", "smart devices", "connected systems",
        "augmented reality", "virtual reality", "mixed reality",
        "autonomous vehicles", "self-driving cars", "transportation systems",
        "smart cities", "urban planning", "sustainable development"
    ]
    
    pattern = random.choice(retrieval_patterns)
    topic = random.choice(retrieval_topics)
    
    prompt = f"{pattern} {topic}"
    
    # Adjust length
    words = prompt.split()
    if len(words) < target_length:
        additional = [
            "from", "the", "database", "or", "knowledge", "base",
            "using", "available", "resources", "and", "sources",
            "in", "the", "system", "or", "repository", "of", "information",
            "through", "available", "channels", "and", "data", "sources",
            "from", "various", "sources", "including", "databases", "and", "archives",
            "using", "search", "engines", "and", "information", "retrieval", "systems",
            "from", "the", "knowledge", "base", "or", "information", "repository",
            "through", "data", "mining", "and", "information", "extraction", "techniques"
        ]
        needed = target_length - len(words)
        # Use random.choices with replacement if needed > available
        if needed <= len(additional):
            words.extend(random.sample(additional, needed))
        else:
            # Sample with replacement if we need more words than available
            words.extend(random.choices(additional, k=needed))
    elif len(words) > target_length:
        words = words[:target_length]
    
    return " ".join(words)


def generate_iteration_prompt(target_length: int, seed: int = None) -> str:
    """Generate a prompt with iterative patterns for iteration head detection."""
    if seed is not None:
        random.seed(seed)
    
    # Create iterative sequences - Expanded
    base_sequences = [
        ["step", "one"], ["step", "two"], ["step", "three"], ["step", "four"], ["step", "five"],
        ["iteration", "1"], ["iteration", "2"], ["iteration", "3"], ["iteration", "4"], ["iteration", "5"],
        ["cycle", "A"], ["cycle", "B"], ["cycle", "C"], ["cycle", "D"], ["cycle", "E"],
        ["round", "first"], ["round", "second"], ["round", "third"], ["round", "fourth"], ["round", "fifth"],
        ["phase", "one"], ["phase", "two"], ["phase", "three"],
        ["stage", "1"], ["stage", "2"], ["stage", "3"], ["stage", "4"],
        ["level", "one"], ["level", "two"], ["level", "three"],
        ["pass", "first"], ["pass", "second"], ["pass", "third"],
        ["run", "1"], ["run", "2"], ["run", "3"], ["run", "4"], ["run", "5"],
        ["batch", "A"], ["batch", "B"], ["batch", "C"],
        ["epoch", "1"], ["epoch", "2"], ["epoch", "3"]
    ]
    
    # Single tokens for filling - Expanded
    single_tokens = [
        "step", "next", "then", "and", "or", "plus",
        "after", "before", "during", "while", "until",
        "also", "further", "more", "again", "repeat",
        "continue", "proceed", "follow", "subsequent"
    ]
    
    tokens = []
    current_length = 0
    iteration_count = 0
    
    while current_length < target_length:
        remaining = target_length - current_length
        
        # If not enough space for a full sequence, use single tokens
        if remaining < 3:
            for _ in range(remaining):
                tokens.append(random.choice(single_tokens))
            break
        
        # Select a base sequence
        base = random.choice(base_sequences)
        
        # Check if we have space for base + number + separator
        needed = len(base) + 1  # base + number
        if current_length + needed <= target_length:
            tokens.extend(base)
            tokens.append(str(iteration_count % 10))
            current_length += needed
            iteration_count += 1
            
            # Add separator if we have space
            if current_length < target_length:
                tokens.append("then")
                current_length += 1
        else:
            # Not enough space, fill with single tokens
            for _ in range(remaining):
                tokens.append(random.choice(single_tokens))
            break
    
    return " ".join(tokens[:target_length])


def generate_prompts(head_type: str, max_length: int = 100, 
                    length_distribution: Dict[int, int] = None,
                    min_length: int = None, interval: int = None, 
                    length_per_interval: int = None) -> List[str]:
    """
    Generate prompts for a specific head type with length distribution.
    
    Args:
        head_type: Type of head to generate prompts for. Options:
                  - "duplicate_token_head"
                  - "induction_head"
                  - "previous_token_head"
                  - "truthfulness"
                  - "retrieval"
                  - "iteration"
        max_length: Maximum prompt length (default: 100)
        length_distribution: Optional custom length distribution dictionary.
                            If None, uses default distribution or generates based on
                            min_length, interval, and length_per_interval.
        min_length: Minimum prompt length (default: None, uses default strategy)
        interval: Interval between lengths (default: None, uses default strategy)
        length_per_interval: Number of prompts per interval length (default: None, uses default strategy)
        
    Returns:
        List of generated prompts
        
    Examples:
        # Default strategy
        >>> prompts = generate_prompts("induction_head", max_length=50)
        
        # Custom: every 10 tokens from 10 to 100, 5 prompts each
        >>> prompts = generate_prompts("induction_head", max_length=100, 
        ...                            min_length=10, interval=10, length_per_interval=5)
    """
    # Get length distribution
    if length_distribution is None:
        length_distribution = get_length_distribution(
            max_length=max_length,
            min_length=min_length,
            interval=interval,
            length_per_interval=length_per_interval
        )
    
    # Map head types to generator functions
    generator_map = {
        "duplicate_token_head": generate_duplicate_token_prompt,
        "induction_head": generate_induction_prompt,
        "previous_token_head": generate_previous_token_prompt,
        "truthfulness": generate_truthfulness_prompt,
        "retrieval": generate_retrieval_prompt,
        "iteration": generate_iteration_prompt
    }
    
    # Validate head type
    if head_type not in generator_map:
        raise ValueError(
            f"Unknown head_type: {head_type}. "
            f"Supported types: {list(generator_map.keys())}"
        )
    
    # Original example prompts from head_recog.py
    original_examples = {
        "duplicate_token_head": [
            "one two three one two three one two three",
            "1 2 3 4 5 1 2 3 4 1 2 3 1 2 3 4 5 6 7",
            "green ideas sleep furiously; green ideas don't sleep furiously"
        ],
        "induction_head": [
            "one two three one two three one two three",
            "1 2 3 4 5 1 2 3 4 1 2 3 1 2 3 4 5 6 7",
            "green ideas sleep furiously; green ideas don't sleep furiously"
        ],
        "previous_token_head": [
            "The head detector feature for TransformerLens allows users to check for various common heads automatically, reducing the cost of discovery.",
            "Machine learning models require careful evaluation to ensure they perform well on unseen data.",
            "Attention mechanisms in transformers allow models to focus on relevant parts of the input sequence."
        ],
        "truthfulness": [],  # No original examples provided
        "retrieval": [],  # No original examples provided
        "iteration": []  # No original examples provided
    }
    
    # Start with original examples (if they fit within max_length)
    all_prompts = []
    if head_type in original_examples:
        for example in original_examples[head_type]:
            example_length = len(example.split())
            if example_length <= max_length:
                all_prompts.append(example)
    
    # Generate additional prompts
    generator_func = generator_map[head_type]
    
    for length, count in sorted(length_distribution.items()):
        for i in range(count):
            seed = hash(f"{head_type}_{length}_{i}") % (2**31)
            prompt = generator_func(length, seed=seed)
            all_prompts.append(prompt)
    
    return all_prompts


def get_prompt_statistics(prompts: List[str]) -> Dict:
    """Get statistics about generated prompts."""
    lengths = [len(prompt.split()) for prompt in prompts]
    
    return {
        "total_prompts": len(prompts),
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
        "length_distribution": dict(sorted(defaultdict(int, {l: lengths.count(l) for l in lengths}).items()))
    }


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("Prompt Generator for Head Detection")
    print("=" * 70)
    
    # Test all head types
    head_types = [
        "duplicate_token_head",
        "induction_head",
        "previous_token_head",
        # "truthfulness",
        # "retrieval",
        # "iteration"
    ]
    
    # Configuration parameters
    max_length = 100
    min_length = 10
    interval = 10
    length_per_interval = 5
    
    for head_type in head_types:
        print(f"\n{'='*70}")
        print(f"Generating prompts for: {head_type}")
        print(f"{'='*70}")
        print(f"Parameters: min_length={min_length}, max_length={max_length}, "
              f"interval={interval}, length_per_interval={length_per_interval}")
        
        prompts = generate_prompts(
            head_type, 
            max_length=max_length,
            min_length=min_length,
            interval=interval,
            length_per_interval=length_per_interval
        )
        stats = get_prompt_statistics(prompts)
        
        print(f"Total prompts generated: {stats['total_prompts']}")
        print(f"Length range: {stats['min_length']} - {stats['max_length']} tokens")
        print(f"Average length: {stats['avg_length']:.2f} tokens")
        print(f"\nFirst 5 prompts:")
        for i, prompt in enumerate(prompts[:5], 1):
            print(f"  {i}. [{len(prompt.split())} tokens] {prompt[:80]}...")
        
        print(f"\nLength distribution (showing first 10):")
        for length, count in list(stats['length_distribution'].items())[:10]:
            print(f"  Length {length}: {count} prompts")
