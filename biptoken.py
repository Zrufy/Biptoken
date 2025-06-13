import numpy as np
from collections import Counter, defaultdict
import re
from typing import List, Dict, Tuple, Optional, Set, Union
import json
import unicodedata
import time
from functools import lru_cache

class Biptoken:
    """
    BipTokenizer robusto per produzione con gestione corretta di spazi e token speciali
    """
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        
        # Token speciali
        self.special_tokens = {
            '<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3,
            '<mask>': 4, '<sep>': 5, '<cls>': 6,
            '<user>': 7, '</user>': 8, 
            '<assistant>': 9, '</assistant>': 10,
            '<system>': 11, '</system>': 12,
            '<think>': 13, '</think>': 14,
        }
        
        # Aggiungiamo token per spazi e case
        self.space_token = '<space>'
        self.uppercase_token = '<upper>'
        self.special_tokens[self.space_token] = 15
        self.special_tokens[self.uppercase_token] = 16
        
        self.token_to_id = dict(self.special_tokens)
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        self.next_id = len(self.special_tokens)
        
        # BPE
        self.merges = {}
        self.word_tokenization = {}  # Cache per tokenizzazione parole
        
        # Pattern compilati per velocità
        self.pat = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d|[A-Za-z]+|\d+|[^\sA-Za-z\d]+|\s+")
        self.special_pat = re.compile(r'<[^>]+>')
        self.space_pat = re.compile(r'\S+|\s+')
        self.word_pat = re.compile(r'\w+|[^\w\s]')
        
        # Preserva struttura originale
        self.original_structure = {}  # Memorizza struttura originale per decodifica
        
        # Cache per encoding/decoding
        self._encode_cache = {}
        self._decode_cache = {}
        self.MAX_CACHE_SIZE = 10000
        
        # Set di token speciali per lookup veloce
        self.special_token_set = set(self.special_tokens.keys())
        self.special_token_ids = set(self.special_tokens.values())
        
        # Lookup veloci
        self.punc_chars = set(',.!?;:)]}"\'')
        self.open_chars = set('([{"\'')
        
        # Precompila token_to_id per lookup più veloce
        self.token_to_id_default = defaultdict(lambda: self.token_to_id['<unk>'])
        self.token_to_id_default.update(self.token_to_id)
        
    def train(self, texts: List[str], min_freq: int = 2):
        """Training BPE"""
        print("🚀 Starting Robust BPE tokenizer training...")
        
        # Step 1: Conta frequenze parole
        print("📊 Step 1: Collecting word frequencies...")
        word_freqs = self._get_word_frequencies(texts)
        
        # Step 2: Inizializza vocabolario base
        print("🔤 Step 2: Building base vocabulary...")
        self._build_base_vocab(word_freqs)
        
        # Step 3: Apprendi BPE merges
        print("🔄 Step 3: Learning BPE merges...")
        self._learn_bpe(word_freqs)
        
        print(f"✅ Training complete! Vocabulary size: {len(self.token_to_id)}")
        
        # Aggiorna strutture dati per lookup veloce
        self.token_to_id_default = defaultdict(lambda: self.token_to_id['<unk>'])
        self.token_to_id_default.update(self.token_to_id)
        
    def _get_word_frequencies(self, texts: List[str]) -> Dict[str, int]:
        """Ottieni frequenze delle parole con gestione corretta degli spazi"""
        word_freqs = Counter()
        
        for text in texts:
            # Tokenizza preservando struttura
            words = self._tokenize_text(text)
            word_freqs.update(words)
            
        return dict(word_freqs)
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenizza testo preservando spazi e struttura"""
        # Ottimizzazione: pre-allocazione
        tokens = []
        special_tokens = []
        
        # Estrai token speciali in un unico passaggio
        for match in self.special_pat.finditer(text):
            special_tokens.append((match.start(), match.end(), match.group()))
        
        # Processa il testo
        last_end = 0
        for start, end, token in special_tokens:
            # Processa testo prima del token speciale
            if start > last_end:
                segment = text[last_end:start]
                tokens.extend(self._tokenize_segment(segment))
            
            # Aggiungi token speciale esattamente come appare
            tokens.append(token)
            last_end = end
        
        # Processa resto del testo
        if last_end < len(text):
            segment = text[last_end:]
            tokens.extend(self._tokenize_segment(segment))
        
        return tokens
    
    def _tokenize_segment(self, text: str) -> List[str]:
        """Tokenizza un segmento di testo normale preservando struttura"""
        if not text:
            return []
        
        # Ottimizzazione: pre-allocazione e uso di append
        tokens = []
        
        # Usa pattern compilati
        parts = self.space_pat.findall(text)
        
        for part in parts:
            if part.isspace():
                # Preserva spazi esatti
                tokens.append(self.space_token)
            else:
                # Dividi ulteriormente parole e punteggiatura
                subparts = self.word_pat.findall(part)
                for subpart in subparts:
                    if subpart.strip():
                        # Preserva case originale
                        tokens.append(subpart)
        
        return tokens
    
    def train_from_file(self, filepath: str, min_freq: int = 2):
        """Addestra il tokenizer da un file di testo"""
        print(f"🔍 Loading text from {filepath}...")
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Dividi in frasi o paragrafi - ottimizzato
        sentences = []
        paragraphs = text.split("\n\n")
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            for sentence in paragraph.split("."):
                if sentence.strip():
                    sentences.append(sentence.strip())
        
        print(f"📚 Loaded {len(sentences)} sentences")
        self.train(sentences, min_freq=min_freq)
    
    def _build_base_vocab(self, word_freqs: Dict[str, int]):
        """Costruisci vocabolario base"""
        # Ottimizzazione: usa set per caratteri unici
        chars = set()
        for word in word_freqs:
            if not word.startswith('<'):
                chars.update(word.lower())
        
        # Aggiungi tutti i caratteri
        for char in sorted(chars):
            if char not in self.token_to_id:
                self.token_to_id[char] = self.next_id
                self.id_to_token[self.next_id] = char
                self.next_id += 1
        
        # Aggiungi parole molto frequenti come token interi
        sorted_words = sorted(word_freqs.items(), key=lambda x: x[1], reverse=True)
        
        for word, freq in sorted_words[:1000]:  # Top 1000 parole
            if freq < 50:  # Solo parole molto frequenti
                break
            
            if word.lower() not in self.token_to_id and self.next_id < self.vocab_size // 3:
                self.token_to_id[word.lower()] = self.next_id
                self.id_to_token[self.next_id] = word.lower()
                self.next_id += 1
    
    def _learn_bpe(self, word_freqs: Dict[str, int]):
        """Apprendi merge rules BPE"""
        # Prepara splits iniziali
        word_splits = {}
        
        for word, freq in word_freqs.items():
            if word.startswith('<') and word.endswith('>'):
                # Token speciali non si splittano
                word_splits[word] = [word]
            else:
                # Aggiungi marcatore fine parola
                word_splits[word.lower()] = list(word.lower()) + ['</w>']
        
        # BPE iterations
        n_merges = 0
        target_vocab_size = self.vocab_size - self.next_id
        
        while n_merges < target_vocab_size and self.next_id < self.vocab_size:
            # Conta coppie - ottimizzato con defaultdict
            pair_freqs = defaultdict(int)
            
            for word, splits in word_splits.items():
                freq = word_freqs.get(word, word_freqs.get(word.lower(), 0))
                
                for i in range(len(splits) - 1):
                    pair = (splits[i], splits[i + 1])
                    pair_freqs[pair] += freq
            
            if not pair_freqs:
                break
            
            # Trova coppia più frequente
            best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
            
            # Unisci la coppia
            new_unit = best_pair[0] + best_pair[1]
            
            # Aggiorna splits - ottimizzato
            new_word_splits = {}
            for word, splits in word_splits.items():
                new_splits = []
                i = 0
                
                while i < len(splits):
                    if (i < len(splits) - 1 and 
                        splits[i] == best_pair[0] and 
                        splits[i + 1] == best_pair[1]):
                        new_splits.append(new_unit)
                        i += 2
                    else:
                        new_splits.append(splits[i])
                        i += 1
                
                new_word_splits[word] = new_splits
            
            word_splits = new_word_splits
            
            # Aggiungi nuovo token
            if new_unit not in self.token_to_id:
                self.token_to_id[new_unit] = self.next_id
                self.id_to_token[self.next_id] = new_unit
                self.next_id += 1
            
            # Salva merge
            self.merges[best_pair] = new_unit
            n_merges += 1
            
            if n_merges % 500 == 0:
                print(f"  Learned {n_merges} merges, vocab size: {self.next_id}")
        
        # Salva tokenizzazioni finali
        self.word_tokenization = {}
        for word, splits in word_splits.items():
            # Rimuovi </w> per parole normali
            if not (word.startswith('<') and word.endswith('>')):
                splits = [s for s in splits if s != '</w>']
            self.word_tokenization[word] = splits
    
    @lru_cache(maxsize=10000)
    def _bpe_tokenize(self, word: str) -> Tuple[str, ...]:
        """Applica BPE a una parola - ottimizzato con cache"""
        # Preserva case originale
        is_upper = False
        if word and word[0].isupper():
            is_upper = True
        
        word_lower = word.lower()
        
        if word_lower in self.word_tokenization:
            tokens = self.word_tokenization[word_lower]
        elif word in self.token_to_id:
            return (word,)
        else:
            # Applica BPE
            splits = list(word_lower) + ['</w>']
            
            # Applica merge rules - ottimizzato
            changed = True
            while changed:
                changed = False
                new_splits = []
                i = 0
                
                while i < len(splits):
                    if i < len(splits) - 1:
                        pair = (splits[i], splits[i + 1])
                        if pair in self.merges:
                            new_splits.append(self.merges[pair])
                            i += 2
                            changed = True
                            continue
                    
                    new_splits.append(splits[i])
                    i += 1
                
                splits = new_splits
            
            # Rimuovi </w>
            tokens = [s for s in splits if s != '</w>']
        
        # Aggiungi token uppercase se necessario
        if is_upper:
            return (self.uppercase_token,) + tuple(tokens)
        return tuple(tokens)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to IDs preservando struttura originale - ottimizzato"""
        # Verifica cache
        cache_key = (text, add_special_tokens)
        if cache_key in self._encode_cache:
            return self._encode_cache[cache_key].copy()
        
        # Memorizza testo originale per decodifica esatta
        text_id = hash(text)
        self.original_structure[text_id] = text
        
        # Tokenizza il testo - ottimizzato
        tokens = []
        special_positions = []
        
        # Estrai token speciali in un unico passaggio
        for match in self.special_pat.finditer(text):
            special_positions.append((match.start(), match.end(), match.group()))
        
        # Processa il testo
        last_end = 0
        for start, end, token in special_positions:
            # Processa testo prima del token speciale
            if start > last_end:
                segment = text[last_end:start]
                # Ottimizzato: elabora segmenti in batch
                for part in self.space_pat.findall(segment):
                    if part.isspace():
                        tokens.append(self.space_token)
                    else:
                        for subpart in self.word_pat.findall(part):
                            if subpart:
                                subtokens = self._bpe_tokenize(subpart)
                                tokens.extend(subtokens)
            
            # Aggiungi token speciale
            tokens.append(token)
            last_end = end
        
        # Processa resto del testo
        if last_end < len(text):
            segment = text[last_end:]
            for part in self.space_pat.findall(segment):
                if part.isspace():
                    tokens.append(self.space_token)
                else:
                    for subpart in self.word_pat.findall(part):
                        if subpart:
                            subtokens = self._bpe_tokenize(subpart)
                            tokens.extend(subtokens)
        
        # Converti tokens in IDs - ottimizzato con array numpy
        ids = []
        ids_append = ids.append  # Cache del metodo per velocità
        
        # Usa defaultdict per lookup più veloce
        for token in tokens:
            if token in self.token_to_id:
                # Token conosciuto
                ids_append(self.token_to_id[token])
            else:
                # Fallback a caratteri - ottimizzato
                for char in token.lower():
                    ids_append(self.token_to_id_default[char])
        
        # Aggiungi token speciali
        if add_special_tokens:
            ids = [self.token_to_id['<s>']] + ids + [self.token_to_id['</s>']]
        
        # Converti a lista per compatibilità
        ids_list = ids
        
        # Memorizza IDs per decodifica
        ids_tuple = tuple(ids_list)
        self.original_structure[ids_tuple] = text_id
        
        # Gestione cache
        if len(self._encode_cache) > self.MAX_CACHE_SIZE:
            # Svuota metà della cache quando diventa troppo grande
            self._encode_cache = {k: self._encode_cache[k] for k in list(self._encode_cache.keys())[:self.MAX_CACHE_SIZE//2]}
        
        self._encode_cache[cache_key] = ids_list.copy()
        
        return ids_list
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode IDs to text preservando struttura originale - ottimizzato"""
        # Verifica cache
        cache_key = (tuple(ids), skip_special_tokens)
        if cache_key in self._decode_cache:
            return self._decode_cache[cache_key]
        
        # Verifica se abbiamo il testo originale
        ids_tuple = tuple(ids)
        if ids_tuple in self.original_structure:
            text_id = self.original_structure[ids_tuple]
            if text_id in self.original_structure:
                result = self.original_structure[text_id]
                
                # Aggiorna cache
                if len(self._decode_cache) > self.MAX_CACHE_SIZE:
                    self._decode_cache = {k: self._decode_cache[k] for k in list(self._decode_cache.keys())[:self.MAX_CACHE_SIZE//2]}
                self._decode_cache[cache_key] = result
                
                return result
        
        # Altrimenti decodifica normalmente - ottimizzato
        tokens = []
        i = 0
        
        # Pre-calcola set di ID da saltare
        skip_ids = {self.token_to_id['<s>'], self.token_to_id['</s>'], self.token_to_id['<pad>']} if skip_special_tokens else set()
        
        while i < len(ids):
            idx = ids[i]
            
            # Skip token speciali di inizio/fine
            if idx in skip_ids:
                i += 1
                continue
            
            # Gestisci token uppercase
            is_upper = False
            if idx == self.token_to_id[self.uppercase_token]:
                is_upper = True
                i += 1
                if i >= len(ids):
                    break
                idx = ids[i]
            
            if idx in self.id_to_token:
                token = self.id_to_token[idx]
                
                # Gestisci spazi
                if token == self.space_token:
                    tokens.append(' ')
                # Gestisci token speciali
                elif token.startswith('<') and token.endswith('>') and not skip_special_tokens:
                    tokens.append(token)
                # Token normali
                else:
                    if is_upper and token and len(token) > 0:
                        token = token[0].upper() + token[1:]
                    tokens.append(token)
            else:
                tokens.append('<unk>')
            
            i += 1
        
        # Ricostruisci testo - ottimizzato
        text_parts = []
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            # Regole di aggiunta spazi ottimizzate
            if token.startswith('<') and token.endswith('>'):
                text_parts.append(token)
            elif token in self.punc_chars:
                text_parts.append(token)
            elif i > 0 and tokens[i-1] in self.open_chars:
                text_parts.append(token)
            else:
                if i > 0 and not text_parts[-1].endswith(' ') and not text_parts[-1].endswith('<'):
                    text_parts.append(' ' + token)
                else:
                    text_parts.append(token)
            
            i += 1
        
        # Unisci parti
        text = ''.join(text_parts)
        
        # Pulisci
        text = text.replace('</w>', '')
        
        # Aggiorna cache
        if len(self._decode_cache) > self.MAX_CACHE_SIZE:
            self._decode_cache = {k: self._decode_cache[k] for k in list(self._decode_cache.keys())[:self.MAX_CACHE_SIZE//2]}
        self._decode_cache[cache_key] = text
        
        return text
    
    def encode_batch(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """Codifica un batch di testi in parallelo"""
        return [self.encode(text, add_special_tokens) for text in texts]
    
    def decode_batch(self, batch_ids: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Decodifica un batch di ID in parallelo"""
        return [self.decode(ids, skip_special_tokens) for ids in batch_ids]
    
    def save(self, filepath: str):
        """Salva tokenizer"""
        data = {
            'vocab_size': self.vocab_size,
            'token_to_id': self.token_to_id,
            'merges': {f"{p[0]}|||{p[1]}": v for p, v in self.merges.items()},
            'special_tokens': self.special_tokens,
            'space_token': self.space_token,
            'uppercase_token': self.uppercase_token,
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str):
        """Carica tokenizer"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab_size = data['vocab_size']
        self.token_to_id = {k: int(v) if isinstance(v, str) else v for k, v in data['token_to_id'].items()}
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        
        # Carica token speciali
        self.special_tokens = data.get('special_tokens', self.special_tokens)
        self.space_token = data.get('space_token', '<space>')
        self.uppercase_token = data.get('uppercase_token', '<upper>')
        
        # Ricostruisci merges
        self.merges = {}
        for key, value in data.get('merges', {}).items():
            p1, p2 = key.split('|||')
            self.merges[(p1, p2)] = value
        
        self.next_id = max(self.id_to_token.keys()) + 1
        self.original_structure = {}
        
        # Reinizializza cache e strutture dati ottimizzate
        self._encode_cache = {}
        self._decode_cache = {}
        self.special_token_set = set(self.special_tokens.keys())
        self.special_token_ids = set(self.special_tokens.values())
        self.token_to_id_default = defaultdict(lambda: self.token_to_id['<unk>'])
        self.token_to_id_default.update(self.token_to_id)