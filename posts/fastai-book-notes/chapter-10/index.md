---
categories:
- ai
- fastai
- notes
- pytorch
date: 2022-3-29
description: Chapter 10 covers text preprocessing and training an RNN for text classification.
hide: false
layout: post
search_exclude: false
comments:
  utterances:
    repo: cj-mills/christianjmills
title: Notes on fastai Book Ch. 10
toc: false

aliases:
- /Notes-on-Fastai-Book-10/

---

* [NLP Deep Dive](#nlp-deep-dive)
* [Text Preprocessing](#text-preprocessing)
* [Training a Text Classifier](#training-a-text-classifier)
* [Disinformation and Language Models](#disinformation-and-language-models)
* [References](#references)

-----

```python
import fastbook
fastbook.setup_book()
```


```python
#hide
from fastbook import *
from IPython.display import display,HTML
```


```python
import inspect
def print_source(obj):
    for line in inspect.getsource(obj).split("\n"):
        print(line)
```



## NLP Deep Dive: RNNs

* In NLP, pretrained models are typically trained on a different type of task than your target task
* A language model is trained to predict the next word in a text (having read the ones before)
* We do not feed the model labes, we just feed it lots of text
* the model uses self-supervised learning to develop an understanding the underlying language of the text
* **Self-Supervised Learning:** training a model using labels that are embedded in the independent variables, rather than requireing external labels
* Self-supervised learning can also be used in other domains
    * [Self-supervised learning and computer vision](https://www.fast.ai/2020/01/13/self_supervised/)
* Self-supervised learning is not usually used for the model that is trained directly
    * used for pretraining a model that is then used for transfer learning
* A pretrained language model is often trained using a different body of text than the one you are targeting
    * can be useful to further pretrain the model on your target body of text
    
#### [Universal Language Model Fine-tunine (ULMFiT)](https://arxiv.org/abs/1801.06146)
* showed that fine-tuning a language model on on the target body of text prior to transfer learning to a classification task, resulted in significantly better predictions



## Text Preprocessing

* can use an approach similar to preparing categorical variables
    1. Make a list of all possible levels of that categorical variable (called the vocab)
        * **Tokenization:** convert the text to a list of words
    2. Replace each level with its index in the vocab
        * **Numericalization:** the process of mapping tokens to numbers
            * List all of the unique words that appear, and convert eachword into a number by looking up its index in the vocab
    3. Create an embedding matrix for this contianing a row for each item in the vocab
        * **Language model data loader creation:** fastai provides an [LMDataLoader](https://docs.fast.ai/text.data.html#LMDataLoader) that automatically handles creating a depdendent variable that is offset from the independent variable by one token. Also handles some important details such as how to shuffle the training data in such a way that the dependent and independent variables maintain their structure as required
    4. Use this embedding matrix as the first layer of a neural network
        * a dedicated embedding matrix can take as inputs the raw vocab indexes created in step 2
        * **Language Model Creation:** need a special kind of model that can handle arbitrarily big or small input lists
* we first concatenate all documents in our dataset into one big long string and split it into words (or tokens)
* our independent variable will be the sequence of words starting with the first word in our long list of words and ending with the second to last.
* our dependent variable will be the sequence of words starting with the second word and ending with the last word
* our vocab will consist of a mix of common words and new words specific to target dataset
* we can use the corresponding rows in the embedding matrix for the pretrained  model and only initialize new rows in the matrix for new words

### Tokenization
**token:** one element of a list created by the tokenization process. 
    * could be a word, part of a word, or a single character

* tokenization is an active field of research, with new and improved tokenizers coming out all the time
  
#### Approaches
* Word-based
    * split a sentence on spaces, as well as applying language-specific rules to try to separate parts of meaning even when there are no spaces
    * punctuation marks are typically split into separate tokens
    * relies on the assumption that spaces provide a useful separation of components of meaning in a sentence
    * some languages don't have spaces or even a well-defined concept of a "word"
* Subword-based
    * split words into smaller parts, based on the most commonly occurring sub-strings
    * Example: "occasion" -> "o c ca sion"
    * handles every human language without needing language specific algorithms to be develop
    * can handle other sequences like genomic sequences or MIDI music notation
* Character-based
    * split a sentence into its individual characters

### Word Tokenization with fastai
* fastai provides a consistent interface to a range of tokenizers in external libraries
* The default English word tokenizer for fastai uses [spaCy](https://spacy.io/)

-----


```python
from fastai.text.all import *
```

-----


```python
URLs.IMDB
```
```text
'https://s3.amazonaws.com/fast-ai-nlp/imdb.tgz'
```

-----


```python
path = untar_data(URLs.IMDB)
path
```
```text
Path('/home/innom-dt/.fastai/data/imdb')
```



#### fastai get_text_files
* [Documentation](https://docs.fast.ai/data.transforms.html#get_text_files)
* Get text files in path recursively, only in folders, if specified.

-----


```python
get_text_files
```
```text
<function fastai.data.transforms.get_text_files(path, recurse=True, folders=None)>
```

-----


```python
print_source(get_text_files)
```
```text
def get_text_files(path, recurse=True, folders=None):
    "Get text files in `path` recursively, only in `folders`, if specified."
    return get_files(path, extensions=['.txt'], recurse=recurse, folders=folders)
```

-----


```python
print_source(get_files)
```
```text
def get_files(path, extensions=None, recurse=True, folders=None, followlinks=True):
    "Get all the files in `path` with optional `extensions`, optionally with `recurse`, only in `folders`, if specified."
    path = Path(path)
    folders=L(folders)
    extensions = setify(extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i,(p,d,f) in enumerate(os.walk(path, followlinks=followlinks)): # returns (dirpath, dirnames, filenames)
            if len(folders) !=0 and i==0: d[:] = [o for o in d if o in folders]
            else:                         d[:] = [o for o in d if not o.startswith('.')]
            if len(folders) !=0 and i==0 and '.' not in folders: continue
            res += _get_files(p, f, extensions)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, f, extensions)
    return L(res)
```

-----


```python
files = get_text_files(path, folders = ['train', 'test', 'unsup'])
```

-----


```python
len(files)
```
```text
100000
```

-----


```python
txt = files[0].open().read(); txt[:75]
```
```text
'This conglomeration fails so miserably on every level that it is difficult '
```



#### fastai SpacyTokenizer
* [Documentation](https://docs.fast.ai/text.core.html#SpacyTokenizer)

-----


```python
WordTokenizer
```
```text
fastai.text.core.SpacyTokenizer
```

-----


```python
print_source(WordTokenizer)
```
```text
class SpacyTokenizer():
    "Spacy tokenizer for `lang`"
    def __init__(self, lang='en', special_toks=None, buf_sz=5000):
        self.special_toks = ifnone(special_toks, defaults.text_spec_tok)
        nlp = spacy.blank(lang)
        for w in self.special_toks: nlp.tokenizer.add_special_case(w, [{ORTH: w}])
        self.pipe,self.buf_sz = nlp.pipe,buf_sz

    def __call__(self, items):
        return (L(doc).attrgot('text') for doc in self.pipe(map(str,items), batch_size=self.buf_sz))
```

-----


```python
first
```
```text
<function fastcore.basics.first(x, f=None, negate=False, **kwargs)>
```

-----


```python
print_source(first)
```
```text
<function first at 0x7fdb8da3de50>
def first(x, f=None, negate=False, **kwargs):
    "First element of `x`, optionally filtered by `f`, or None if missing"
    x = iter(x)
    if f: x = filter_ex(x, f=f, negate=negate, gen=True, **kwargs)
    return next(x, None)
```

-----


```python
coll_repr
```
```text
<function fastcore.foundation.coll_repr(c, max_n=10)>
```

-----


```python
print_source(coll_repr)
```
```text
def coll_repr(c, max_n=10):
    "String repr of up to `max_n` items of (possibly lazy) collection `c`"
    return f'(#{len(c)}) [' + ','.join(itertools.islice(map(repr,c), max_n)) + (
        '...' if len(c)>max_n else '') + ']'
```

-----


```python
spacy = WordTokenizer()
```

-----


```python
spacy.buf_sz
```
```text
5000
```

-----


```python
spacy.pipe
```
```text
<bound method Language.pipe of <spacy.lang.en.English object at 0x7fdb6545f1c0>>
```

-----


```python
spacy.special_toks
```
```text
['xxunk',
 'xxpad',
 'xxbos',
 'xxeos',
 'xxfld',
 'xxrep',
 'xxwrep',
 'xxup',
 'xxmaj']
```

-----


```python
# Wrap text in a list before feeding it to the tokenizer
toks = first(spacy([txt]))
print(coll_repr(toks, 30))
```
```text
(#174) ['This','conglomeration','fails','so','miserably','on','every','level','that','it','is','difficult','to','decide','what','to','say','.','It','does',"n't",'merit','one','line',',','much','less','ten',',','but'...]
```

-----

```python
first(spacy(['The U.S. dollar $1 is $1.00.']))
```
```text
(#9) ['The','U.S.','dollar','$','1','is','$','1.00','.']
```

-----


```python
Tokenizer
```
```text
fastai.text.core.Tokenizer
```

-----


```python
print_source(Tokenizer)
```
```text
class Tokenizer(Transform):
    "Provides a consistent `Transform` interface to tokenizers operating on `DataFrame`s and folders"
    input_types = (str, list, L, tuple, Path)
    def __init__(self, tok, rules=None, counter=None, lengths=None, mode=None, sep=' '):
        if isinstance(tok,type): tok=tok()
        store_attr('tok,counter,lengths,mode,sep')
        self.rules = defaults.text_proc_rules if rules is None else rules

    @classmethod
    @delegates(tokenize_df, keep=True)
    def from_df(cls, text_cols, tok=None, rules=None, sep=' ', **kwargs):
        if tok is None: tok = WordTokenizer()
        res = cls(tok, rules=rules, mode='df')
        res.kwargs,res.train_setup = merge({'tok': tok}, kwargs),False
        res.text_cols,res.sep = text_cols,sep
        return res

    @classmethod
    @delegates(tokenize_folder, keep=True)
    def from_folder(cls, path, tok=None, rules=None, **kwargs):
        path = Path(path)
        if tok is None: tok = WordTokenizer()
        output_dir = tokenize_folder(path, tok=tok, rules=rules, **kwargs)
        res = cls(tok, counter=load_pickle(output_dir/fn_counter_pkl),
                  lengths=load_pickle(output_dir/fn_lengths_pkl), rules=rules, mode='folder')
        res.path,res.output_dir = path,output_dir
        return res

    def setups(self, dsets):
        if not self.mode == 'df' or not isinstance(dsets.items, pd.DataFrame): return
        dsets.items,count = tokenize_df(dsets.items, self.text_cols, rules=self.rules, **self.kwargs)
        if self.counter is None: self.counter = count
        return dsets

    def encodes(self, o:Path):
        if self.mode=='folder' and str(o).startswith(str(self.path)):
            tok = self.output_dir/o.relative_to(self.path)
            return L(tok.read_text(encoding='UTF-8').split(' '))
        else: return self._tokenize1(o.read_text())

    def encodes(self, o:str): return self._tokenize1(o)
    def _tokenize1(self, o): return first(self.tok([compose(*self.rules)(o)]))

    def get_lengths(self, items):
        if self.lengths is None: return None
        if self.mode == 'df':
            if isinstance(items, pd.DataFrame) and 'text_lengths' in items.columns: return items['text_length'].values
        if self.mode == 'folder':
            try:
                res = [self.lengths[str(Path(i).relative_to(self.path))] for i in items]
                if len(res) == len(items): return res
            except: return None

    def decodes(self, o): return TitledStr(self.sep.join(o))
```

-----


```python
tkn = Tokenizer(spacy)
print(coll_repr(tkn(txt), 31))
```
```text
(#177) ['xxbos','xxmaj','this','conglomeration','fails','so','miserably','on','every','level','that','it','is','difficult','to','decide','what','to','say','.','xxmaj','it','does',"n't",'merit','one','line',',','much','less','ten'...]
```


#### Special Tokens
* tokens that start with `xx` are special tokens
* designed to make it easier for a model to recognize the important parts of a sentence
* `xxbos`: Indicates the beginning of a text
* `xxmaj`: Indicates the next word begins with a capital letter (since everything is made lowercase)
* `xxunk`: Indicates the next word is unknown

#### Preprocessing Rules


```python
defaults.text_proc_rules
```
```text
[<function fastai.text.core.fix_html(x)>,
 <function fastai.text.core.replace_rep(t)>,
 <function fastai.text.core.replace_wrep(t)>,
 <function fastai.text.core.spec_add_spaces(t)>,
 <function fastai.text.core.rm_useless_spaces(t)>,
 <function fastai.text.core.replace_all_caps(t)>,
 <function fastai.text.core.replace_maj(t)>,
 <function fastai.text.core.lowercase(t, add_bos=True, add_eos=False)>]
```

-----


```python
for rule in defaults.text_proc_rules:
    print_source(rule)
```
```text
def fix_html(x):
    "Various messy things we've seen in documents"
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace('nbsp;', ' ').replace(
        '#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace('<br />', "\n").replace(
        '\\"', '"').replace('<unk>',UNK).replace(' @.@ ','.').replace(' @-@ ','-').replace('...',' …')
    return html.unescape(x)

def replace_rep(t):
    "Replace repetitions at the character level: cccc -- TK_REP 4 c"
    def _replace_rep(m):
        c,cc = m.groups()
        return f' {TK_REP} {len(cc)+1} {c} '
    return _re_rep.sub(_replace_rep, t)

def replace_wrep(t):
    "Replace word repetitions: word word word word -- TK_WREP 4 word"
    def _replace_wrep(m):
        c,cc,e = m.groups()
        return f' {TK_WREP} {len(cc.split())+2} {c} {e}'
    return _re_wrep.sub(_replace_wrep, t)

def spec_add_spaces(t):
    "Add spaces around / and #"
    return _re_spec.sub(r' \1 ', t)

def rm_useless_spaces(t):
    "Remove multiple spaces"
    return _re_space.sub(' ', t)

def replace_all_caps(t):
    "Replace tokens in ALL CAPS by their lower version and add `TK_UP` before."
    def _replace_all_caps(m):
        tok = f'{TK_UP} ' if len(m.groups()[1]) > 1 else ''
        return f"{m.groups()[0]}{tok}{m.groups()[1].lower()}"
    return _re_all_caps.sub(_replace_all_caps, t)

def replace_maj(t):
    "Replace tokens in Sentence Case by their lower version and add `TK_MAJ` before."
    def _replace_maj(m):
        tok = f'{TK_MAJ} ' if len(m.groups()[1]) > 1 else ''
        return f"{m.groups()[0]}{tok}{m.groups()[1].lower()}"
    return _re_maj.sub(_replace_maj, t)

def lowercase(t, add_bos=True, add_eos=False):
    "Converts `t` to lowercase"
    return (f'{BOS} ' if add_bos else '') + t.lower().strip() + (f' {EOS}' if add_eos else '')
```



#### Postprocessing Rules


```python
for rule in defaults.text_postproc_rules:
    print_source(rule)
```
```text
def replace_space(t):
    "Replace embedded spaces in a token with unicode line char to allow for split/join"
    return t.replace(' ', '_')
```

-----


```python
coll_repr(tkn('&copy;   Fast.ai www.fast.ai/INDEX'), 31)
```
```text
"(#11) ['xxbos','©','xxmaj','fast.ai','xxrep','3','w','.fast.ai','/','xxup','index']"
```



### Subword Tokenization

#### Process
1. Analyze a corpus of documents to find the most commonly occurring groups of letters. These become the vocab
2. Tokenize the corpus using this vocab of subword units.

-----


```python
txts = L(o.open().read() for o in files[:2000])
```

-----


```python
SubwordTokenizer
```
```text
fastai.text.core.SentencePieceTokenizer
```

-----


```python
print_source(SubwordTokenizer)
```
```text
class SentencePieceTokenizer():#TODO: pass the special tokens symbol to sp
    "SentencePiece tokenizer for `lang`"
    def __init__(self, lang='en', special_toks=None, sp_model=None, vocab_sz=None, max_vocab_sz=30000,
                 model_type='unigram', char_coverage=None, cache_dir='tmp'):
        try: from sentencepiece import SentencePieceTrainer,SentencePieceProcessor
        except ImportError:
            raise Exception('sentencepiece module is missing: run `pip install sentencepiece!=0.1.90,!=0.1.91`')
        self.sp_model,self.cache_dir = sp_model,Path(cache_dir)
        self.vocab_sz,self.max_vocab_sz,self.model_type = vocab_sz,max_vocab_sz,model_type
        self.char_coverage = ifnone(char_coverage, 0.99999 if lang in eu_langs else 0.9998)
        self.special_toks = ifnone(special_toks, defaults.text_spec_tok)
        if sp_model is None: self.tok = None
        else:
            self.tok = SentencePieceProcessor()
            self.tok.Load(str(sp_model))
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_vocab_sz(self, raw_text_path):
        cnt = Counter()
        with open(raw_text_path, 'r') as f:
            for line in f.readlines():
                cnt.update(line.split())
                if len(cnt)//4 > self.max_vocab_sz: return self.max_vocab_sz
        res = len(cnt)//4
        while res%8 != 0: res+=1
        return max(res,29)

    def train(self, raw_text_path):
        "Train a sentencepiece tokenizer on `texts` and save it in `path/tmp_dir`"
        from sentencepiece import SentencePieceTrainer
        vocab_sz = self._get_vocab_sz(raw_text_path) if self.vocab_sz is None else self.vocab_sz
        spec_tokens = ['\u2581'+s for s in self.special_toks]
        SentencePieceTrainer.Train(" ".join([
            f"--input={raw_text_path} --vocab_size={vocab_sz} --model_prefix={self.cache_dir/'spm'}",
            f"--character_coverage={self.char_coverage} --model_type={self.model_type}",
            f"--unk_id={len(spec_tokens)} --pad_id=-1 --bos_id=-1 --eos_id=-1 --minloglevel=2",
            f"--user_defined_symbols={','.join(spec_tokens)} --hard_vocab_limit=false"]))
        raw_text_path.unlink()
        return self.cache_dir/'spm.model'

    def setup(self, items, rules=None):
        from sentencepiece import SentencePieceProcessor
        if rules is None: rules = []
        if self.tok is not None: return {'sp_model': self.sp_model}
        raw_text_path = self.cache_dir/'texts.out'
        with open(raw_text_path, 'w') as f:
            for t in progress_bar(maps(*rules, items), total=len(items), leave=False):
                f.write(f'{t}\n')
        sp_model = self.train(raw_text_path)
        self.tok = SentencePieceProcessor()
        self.tok.Load(str(sp_model))
        return {'sp_model': sp_model}

    def __call__(self, items):
        if self.tok is None: self.setup(items)
        for t in items: yield self.tok.EncodeAsPieces(t)
```



#### sentencepiece tokenizer
* [GitHub Repository](https://github.com/google/sentencepiece)
* Unsupervised text tokenizer for Neural Network-based text generation. 

-----


```python
def subword(sz):
    # Initialize a tokenizer with the desired vocab size
    sp = SubwordTokenizer(vocab_sz=sz)
    # Generate vocab based on target body of text
    sp.setup(txts)
    return ' '.join(first(sp([txt]))[:40])
```

#### Picking a vocab size
* * provides an easy way to scale between character tokenization and word tokenization
* a smaller vocab size results in each token representing fewer characters
* an overly large vocab size results in most common words ending up in the vocab
    * fewer tokens per sentence
    * faster training
    * less memory
    * less state for the model to remember
    * larger embedding matrices which require more data to learn

-----


```python
# Use a vocab size of 1000
subword(1000)
```
```text
sentencepiece_trainer.cc(177) LOG(INFO) Running command: --input=tmp/texts.out --vocab_size=1000 --model_prefix=tmp/spm --character_coverage=0.99999 --model_type=unigram --unk_id=9 --pad_id=-1 --bos_id=-1 --eos_id=-1 --minloglevel=2 --user_defined_symbols=▁xxunk,▁xxpad,▁xxbos,▁xxeos,▁xxfld,▁xxrep,▁xxwrep,▁xxup,▁xxmaj --hard_vocab_limit=false
```



```text
"▁This ▁con g lo m er ation ▁fail s ▁so ▁mis er ably ▁on ▁every ▁level ▁that ▁it ▁is ▁di ff ic ul t ▁to ▁decide ▁what ▁to ▁say . ▁It ▁doesn ' t ▁me ri t ▁one ▁line ,"
```



**Note:** The **`_`** character represents a space character in the original text


```python
# Use a vocab size of 200
subword(200)
```
```text
'▁ T h i s ▁c on g l o m er at ion ▁f a i l s ▁ s o ▁ m i s er a b ly ▁on ▁ e v er y ▁ le ve l'
```

-----


```python
# Use a vocab size of 10000
subword(10000)
```
```text
"▁This ▁con g l ome ration ▁fails ▁so ▁miserably ▁on ▁every ▁level ▁that ▁it ▁is ▁difficult ▁to ▁decide ▁what ▁to ▁say . ▁It ▁doesn ' t ▁merit ▁one ▁line , ▁much ▁less ▁ten , ▁but ▁to ▁adhere ▁to ▁the ▁rules"
```



### Numericalization with fastai


```python
toks = tkn(txt)
print(coll_repr(tkn(txt), 31))
```
```text
(#177) ['xxbos','xxmaj','this','conglomeration','fails','so','miserably','on','every','level','that','it','is','difficult','to','decide','what','to','say','.','xxmaj','it','does',"n't",'merit','one','line',',','much','less','ten'...]
```

-----

```python
toks200 = txts[:200].map(tkn)
toks200[0]
```
```text
(#177) ['xxbos','xxmaj','this','conglomeration','fails','so','miserably','on','every','level'...]
```



#### fastai Numericalize
* [Documentation](https://docs.fast.ai/text.data.html#Numericalize)

-----


```python
Numericalize
```
```text
fastai.text.data.Numericalize
```

-----


```python
print_source(Numericalize)
```
```text
class Numericalize(Transform):
    "Reversible transform of tokenized texts to numericalized ids"
    def __init__(self, vocab=None, min_freq=3, max_vocab=60000, special_toks=None):
        store_attr('vocab,min_freq,max_vocab,special_toks')
        self.o2i = None if vocab is None else defaultdict(int, {v:k for k,v in enumerate(vocab)})

    def setups(self, dsets):
        if dsets is None: return
        if self.vocab is None:
            count = dsets.counter if getattr(dsets, 'counter', None) is not None else Counter(p for o in dsets for p in o)
            if self.special_toks is None and hasattr(dsets, 'special_toks'):
                self.special_toks = dsets.special_toks
            self.vocab = make_vocab(count, min_freq=self.min_freq, max_vocab=self.max_vocab, special_toks=self.special_toks)
            self.o2i = defaultdict(int, {v:k for k,v in enumerate(self.vocab) if v != 'xxfake'})

    def encodes(self, o): return TensorText(tensor([self.o2i  [o_] for o_ in o]))
    def decodes(self, o): return L(self.vocab[o_] for o_ in o)
```

-----


```python
num = Numericalize()
# Generate vocab
num.setup(toks200)
coll_repr(num.vocab,20)
```
```text
"(#1992) ['xxunk','xxpad','xxbos','xxeos','xxfld','xxrep','xxwrep','xxup','xxmaj','the','.',',','a','and','of','to','is','i','it','this'...]"
```

-----


```python
TensorText
```
```text
fastai.text.data.TensorText
```

-----


```python
print_source(TensorText)
```
```text
class TensorText(TensorBase):   pass
```

-----


```python
nums = num(toks)[:20]; nums
```
```text
TensorText([   2,    8,   19,    0,  585,   51, 1190,   36,  166,  586,   21,   18,   16,    0,   15,  663,   67,   15,  140,   10])
```

-----


```python
' '.join(num.vocab[o] for o in nums)
```
```text
'xxbos xxmaj this xxunk fails so miserably on every level that it is xxunk to decide what to say .'
```



**Note:** Special rules tokens appear first followed by tokens in order of frequency



### Putting Our Texts into Batches for a Language Model


```python
stream = "In this chapter, we will go back over the example of classifying movie reviews we studied in chapter 1 and dig deeper under the surface. First we will look at the processing steps necessary to convert text into numbers and how to customize it. By doing this, we'll have another example of the PreProcessor used in the data block API.\nThen we will study how we build a language model and train it for a while."
tokens = tkn(stream)
tokens
```
```text
(#90) ['xxbos','xxmaj','in','this','chapter',',','we','will','go','back'...]
```

-----


```python
# Visualize 6 batches of 15 tokens
bs,seq_len = 6,15
d_tokens = np.array([tokens[i*seq_len:(i+1)*seq_len] for i in range(bs)])
df = pd.DataFrame(d_tokens)
display(HTML(df.to_html(index=False,header=None)))
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <td>xxbos</td>
      <td>xxmaj</td>
      <td>in</td>
      <td>this</td>
      <td>chapter</td>
      <td>,</td>
      <td>we</td>
      <td>will</td>
      <td>go</td>
      <td>back</td>
      <td>over</td>
      <td>the</td>
      <td>example</td>
      <td>of</td>
      <td>classifying</td>
    </tr>
    <tr>
      <td>movie</td>
      <td>reviews</td>
      <td>we</td>
      <td>studied</td>
      <td>in</td>
      <td>chapter</td>
      <td>1</td>
      <td>and</td>
      <td>dig</td>
      <td>deeper</td>
      <td>under</td>
      <td>the</td>
      <td>surface</td>
      <td>.</td>
      <td>xxmaj</td>
    </tr>
    <tr>
      <td>first</td>
      <td>we</td>
      <td>will</td>
      <td>look</td>
      <td>at</td>
      <td>the</td>
      <td>processing</td>
      <td>steps</td>
      <td>necessary</td>
      <td>to</td>
      <td>convert</td>
      <td>text</td>
      <td>into</td>
      <td>numbers</td>
      <td>and</td>
    </tr>
    <tr>
      <td>how</td>
      <td>to</td>
      <td>customize</td>
      <td>it</td>
      <td>.</td>
      <td>xxmaj</td>
      <td>by</td>
      <td>doing</td>
      <td>this</td>
      <td>,</td>
      <td>we</td>
      <td>'ll</td>
      <td>have</td>
      <td>another</td>
      <td>example</td>
    </tr>
    <tr>
      <td>of</td>
      <td>the</td>
      <td>preprocessor</td>
      <td>used</td>
      <td>in</td>
      <td>the</td>
      <td>data</td>
      <td>block</td>
      <td>xxup</td>
      <td>api</td>
      <td>.</td>
      <td>\n</td>
      <td>xxmaj</td>
      <td>then</td>
      <td>we</td>
    </tr>
    <tr>
      <td>will</td>
      <td>study</td>
      <td>how</td>
      <td>we</td>
      <td>build</td>
      <td>a</td>
      <td>language</td>
      <td>model</td>
      <td>and</td>
      <td>train</td>
      <td>it</td>
      <td>for</td>
      <td>a</td>
      <td>while</td>
      <td>.</td>
    </tr>
  </tbody>
</table>
</div>
-----

```python
# 6 batches of 5 tokens
bs,seq_len = 6,5
d_tokens = np.array([tokens[i*15:i*15+seq_len] for i in range(bs)])
df = pd.DataFrame(d_tokens)
display(HTML(df.to_html(index=False,header=None)))
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <td>xxbos</td>
      <td>xxmaj</td>
      <td>in</td>
      <td>this</td>
      <td>chapter</td>
    </tr>
    <tr>
      <td>movie</td>
      <td>reviews</td>
      <td>we</td>
      <td>studied</td>
      <td>in</td>
    </tr>
    <tr>
      <td>first</td>
      <td>we</td>
      <td>will</td>
      <td>look</td>
      <td>at</td>
    </tr>
    <tr>
      <td>how</td>
      <td>to</td>
      <td>customize</td>
      <td>it</td>
      <td>.</td>
    </tr>
    <tr>
      <td>of</td>
      <td>the</td>
      <td>preprocessor</td>
      <td>used</td>
      <td>in</td>
    </tr>
    <tr>
      <td>will</td>
      <td>study</td>
      <td>how</td>
      <td>we</td>
      <td>build</td>
    </tr>
  </tbody>
</table>
</div>
-----

```python
bs,seq_len = 6,5
d_tokens = np.array([tokens[i*15+seq_len:i*15+2*seq_len] for i in range(bs)])
df = pd.DataFrame(d_tokens)
display(HTML(df.to_html(index=False,header=None)))
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <td>,</td>
      <td>we</td>
      <td>will</td>
      <td>go</td>
      <td>back</td>
    </tr>
    <tr>
      <td>chapter</td>
      <td>1</td>
      <td>and</td>
      <td>dig</td>
      <td>deeper</td>
    </tr>
    <tr>
      <td>the</td>
      <td>processing</td>
      <td>steps</td>
      <td>necessary</td>
      <td>to</td>
    </tr>
    <tr>
      <td>xxmaj</td>
      <td>by</td>
      <td>doing</td>
      <td>this</td>
      <td>,</td>
    </tr>
    <tr>
      <td>the</td>
      <td>data</td>
      <td>block</td>
      <td>xxup</td>
      <td>api</td>
    </tr>
    <tr>
      <td>a</td>
      <td>language</td>
      <td>model</td>
      <td>and</td>
      <td>train</td>
    </tr>
  </tbody>
</table>
</div>
-----

```python
bs,seq_len = 6,5
d_tokens = np.array([tokens[i*15+10:i*15+15] for i in range(bs)])
df = pd.DataFrame(d_tokens)
display(HTML(df.to_html(index=False,header=None)))
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <td>over</td>
      <td>the</td>
      <td>example</td>
      <td>of</td>
      <td>classifying</td>
    </tr>
    <tr>
      <td>under</td>
      <td>the</td>
      <td>surface</td>
      <td>.</td>
      <td>xxmaj</td>
    </tr>
    <tr>
      <td>convert</td>
      <td>text</td>
      <td>into</td>
      <td>numbers</td>
      <td>and</td>
    </tr>
    <tr>
      <td>we</td>
      <td>'ll</td>
      <td>have</td>
      <td>another</td>
      <td>example</td>
    </tr>
    <tr>
      <td>.</td>
      <td>\n</td>
      <td>xxmaj</td>
      <td>then</td>
      <td>we</td>
    </tr>
    <tr>
      <td>it</td>
      <td>for</td>
      <td>a</td>
      <td>while</td>
      <td>.</td>
    </tr>
  </tbody>
</table>
</div>
-----

```python
nums200 = toks200.map(num)
```

-----


```python
LMDataLoader
```
```text
fastai.text.data.LMDataLoader
```

-----


```python
print_source(LMDataLoader)
```
```text
@delegates()
class LMDataLoader(TfmdDL):
    "A `DataLoader` suitable for language modeling"
    def __init__(self, dataset, lens=None, cache=2, bs=64, seq_len=72, num_workers=0, **kwargs):
        self.items = ReindexCollection(dataset, cache=cache, tfm=_maybe_first)
        self.seq_len = seq_len
        if lens is None: lens = _get_lengths(dataset)
        if lens is None: lens = [len(o) for o in self.items]
        self.lens = ReindexCollection(lens, idxs=self.items.idxs)
        # The "-1" is to allow for final label, we throw away the end that's less than bs
        corpus = round_multiple(sum(lens)-1, bs, round_down=True)
        self.bl = corpus//bs #bl stands for batch length
        self.n_batches = self.bl//(seq_len) + int(self.bl%seq_len!=0)
        self.last_len = self.bl - (self.n_batches-1)*seq_len
        self.make_chunks()
        super().__init__(dataset=dataset, bs=bs, num_workers=num_workers, **kwargs)
        self.n = self.n_batches*bs

    def make_chunks(self): self.chunks = Chunks(self.items, self.lens)
    def shuffle_fn(self,idxs):
        self.items.shuffle()
        self.make_chunks()
        return idxs

    def create_item(self, seq):
        if seq is None: seq = 0
        if seq>=self.n: raise IndexError
        sl = self.last_len if seq//self.bs==self.n_batches-1 else self.seq_len
        st = (seq%self.bs)*self.bl + (seq//self.bs)*self.seq_len
        txt = self.chunks[st : st+sl+1]
        return LMTensorText(txt[:-1]),txt[1:]

    @delegates(TfmdDL.new)
    def new(self, dataset=None, seq_len=None, **kwargs):
        lens = self.lens.coll if dataset is None else None
        seq_len = self.seq_len if seq_len is None else seq_len
        return super().new(dataset=dataset, lens=lens, seq_len=seq_len, **kwargs)
```



**Note:** The order of separate documents is shuffled, not the order of the words inside them.


```python
dl = LMDataLoader(nums200)
```

-----


```python
x,y = first(dl)
x.shape,y.shape
```
```text
(torch.Size([64, 72]), torch.Size([64, 72]))
```

-----


```python
' '.join(num.vocab[o] for o in x[0][:20])
```
```text
'xxbos xxmaj this xxunk fails so miserably on every level that it is xxunk to decide what to say .'
```

-----


```python
' '.join(num.vocab[o] for o in y[0][:20])
```
```text
'xxmaj this xxunk fails so miserably on every level that it is xxunk to decide what to say . xxmaj'
```



**Note:** The dependent variable is offset by one token, since the goal is to predict the next token in the sequence.



## Training a Text Classifier
1. Fine-tune a language model pretrained on a standard corpus like Wikipedia on a target dataset
2. Use the fine-tuned model to train a classifier

### Language Model Using DataBlock
* fastai automatically handles tokenization and numericalization when `TextBlock` is passed to `DataBlock`
* fastai saves the tokenized documents in a temporary fodler, so it does not have to tokenize them more than once
* fastai runs multiple tokenization processes in parallel

-----


```python
TextBlock
```
```text
fastai.text.data.TextBlock
```

-----


```python
print_source(TextBlock)
```
```text
class TextBlock(TransformBlock):
    "A `TransformBlock` for texts"
    @delegates(Numericalize.__init__)
    def __init__(self, tok_tfm, vocab=None, is_lm=False, seq_len=72, backwards=False, **kwargs):
        type_tfms = [tok_tfm, Numericalize(vocab, **kwargs)]
        if backwards: type_tfms += [reverse_text]
        return super().__init__(type_tfms=type_tfms,
                                dl_type=LMDataLoader if is_lm else SortedDL,
                                dls_kwargs={'seq_len': seq_len} if is_lm else {'before_batch': Pad_Chunk(seq_len=seq_len)})

    @classmethod
    @delegates(Tokenizer.from_df, keep=True)
    def from_df(cls, text_cols, vocab=None, is_lm=False, seq_len=72, backwards=False, min_freq=3, max_vocab=60000, **kwargs):
        "Build a `TextBlock` from a dataframe using `text_cols`"
        return cls(Tokenizer.from_df(text_cols, **kwargs), vocab=vocab, is_lm=is_lm, seq_len=seq_len,
                   backwards=backwards, min_freq=min_freq, max_vocab=max_vocab)

    @classmethod
    @delegates(Tokenizer.from_folder, keep=True)
    def from_folder(cls, path, vocab=None, is_lm=False, seq_len=72, backwards=False, min_freq=3, max_vocab=60000, **kwargs):
        "Build a `TextBlock` from a `path`"
        return cls(Tokenizer.from_folder(path, **kwargs), vocab=vocab, is_lm=is_lm, seq_len=seq_len,
                   backwards=backwards, min_freq=min_freq, max_vocab=max_vocab)
```

-----


```python
# Define how to get dataset items
get_imdb = partial(get_text_files, folders=['train', 'test', 'unsup'])

dls_lm = DataBlock(
    blocks=TextBlock.from_folder(path, is_lm=True),
    get_items=get_imdb, splitter=RandomSplitter(0.1)
).dataloaders(path, path=path, bs=128, seq_len=80)
```

-----


```python
dls_lm.show_batch(max_n=2)
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>text_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>xxbos i think xxmaj dark xxmaj angel is great ! xxmaj first season was excellent , and had a good plot . xxmaj with xxunk xxmaj alba ) as an escaped xxup xxunk , manticore creation , trying to adapt to a normal life , but still " saving the world " . xxmaj and being hunted by manticore throughout the season which gives the series some extra spice . \n\n xxmaj the second season though suddenly became a bit</td>
      <td>i think xxmaj dark xxmaj angel is great ! xxmaj first season was excellent , and had a good plot . xxmaj with xxunk xxmaj alba ) as an escaped xxup xxunk , manticore creation , trying to adapt to a normal life , but still " saving the world " . xxmaj and being hunted by manticore throughout the season which gives the series some extra spice . \n\n xxmaj the second season though suddenly became a bit odd</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cheating boyfriend xxmaj nick xxmaj gordon planning to drop her for the much younger and sexier xxmaj sally xxmaj higgins . xxmaj sally 's boyfriend xxmaj jerry had earlier participated in a payroll robbery with xxmaj nick where he and a security guard were shot and killed . xxmaj now seeing that there 's a future , in crime , for her with xxmaj nick xxmaj sally willingly replaced xxmaj mimi as xxmaj nick 's new squeeze . xxmaj mad</td>
      <td>boyfriend xxmaj nick xxmaj gordon planning to drop her for the much younger and sexier xxmaj sally xxmaj higgins . xxmaj sally 's boyfriend xxmaj jerry had earlier participated in a payroll robbery with xxmaj nick where he and a security guard were shot and killed . xxmaj now seeing that there 's a future , in crime , for her with xxmaj nick xxmaj sally willingly replaced xxmaj mimi as xxmaj nick 's new squeeze . xxmaj mad as</td>
    </tr>
  </tbody>
</table>
</div>


### Fine-Tuning the Language Model
1. Use embeddings to convert the integer word indices into activations that we can use for our neural network
2. Feed those embeddings to a Recurrent Neural Nerwork (RNN), using an architecture called AWD-LSTM
* This process is handled automatically inside [language_model_learner](https://docs.fast.ai/text.learner.html#language_model_learner)

-----


```python
language_model_learner
```
```text
<function fastai.text.learner.language_model_learner(dls, arch, config=None, drop_mult=1.0, backwards=False, pretrained=True, pretrained_fnames=None, loss_func=None, opt_func=<function Adam at 0x7fdb8123e430>, lr=0.001, splitter=<function trainable_params at 0x7fdb8d9171f0>, cbs=None, metrics=None, path=None, model_dir='models', wd=None, wd_bn_bias=False, train_bn=True, moms=(0.95, 0.85, 0.95))>
```

-----


```python
print_source(language_model_learner)
```
```text
@delegates(Learner.__init__)
def language_model_learner(dls, arch, config=None, drop_mult=1., backwards=False, pretrained=True, pretrained_fnames=None, **kwargs):
    "Create a `Learner` with a language model from `dls` and `arch`."
    vocab = _get_text_vocab(dls)
    model = get_language_model(arch, len(vocab), config=config, drop_mult=drop_mult)
    meta = _model_meta[arch]
    learn = LMLearner(dls, model, loss_func=CrossEntropyLossFlat(), splitter=meta['split_lm'], **kwargs)
    url = 'url_bwd' if backwards else 'url'
    if pretrained or pretrained_fnames:
        if pretrained_fnames is not None:
            fnames = [learn.path/learn.model_dir/f'{fn}.{ext}' for fn,ext in zip(pretrained_fnames, ['pth', 'pkl'])]
        else:
            if url not in meta:
                warn("There are no pretrained weights for that architecture yet!")
                return learn
            model_path = untar_data(meta[url] , c_key='model')
            try: fnames = [list(model_path.glob(f'*.{ext}'))[0] for ext in ['pth', 'pkl']]
            except IndexError: print(f'The model in {model_path} is incomplete, download again'); raise
        learn = learn.load_pretrained(*fnames)
    return learn
```

-----


```python
learn = language_model_learner(
    dls_lm, AWD_LSTM, drop_mult=0.3, 
    metrics=[accuracy, Perplexity()]).to_fp16()
```

#### Perplexity Metric
* the exponential of the loss (i.e. `torch.exp(cross_entropy)`)
* often used in NLP for language models

-----


```python
Perplexity
```
```text
fastai.metrics.Perplexity
```

-----


```python
print_source(Perplexity)
```
```text
class Perplexity(AvgLoss):
    "Perplexity (exponential of cross-entropy loss) for Language Models"
    @property
    def value(self): return torch.exp(self.total/self.count) if self.count != 0 else None
    @property
    def name(self):  return "perplexity"
```

-----


```python
# Train only the embeddings (the only part of the model that contains randomly initialize weights)
learn.fit_one_cycle(1, 2e-2)
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>perplexity</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>4.011688</td>
      <td>3.904507</td>
      <td>0.300504</td>
      <td>49.625618</td>
      <td>09:21</td>
    </tr>
  </tbody>
</table>
</div>


### Saving and Loading Models


```python
learn.save
```
```text
<bound method Learner.save of <fastai.text.learner.LMLearner object at 0x7fdb64fd4190>>
```

-----


```python
print_source(learn.save)
```
```text
@patch
@delegates(save_model)
def save(self:Learner, file, **kwargs):
    "Save model and optimizer state (if `with_opt`) to `self.path/self.model_dir/file`"
    file = join_path_file(file, self.path/self.model_dir, ext='.pth')
    save_model(file, self.model, getattr(self,'opt',None), **kwargs)
    return file
```

-----


```python
save_model
```
```text
<function fastai.learner.save_model(file, model, opt, with_opt=True, pickle_protocol=2)>
```

-----


```python
print_source(save_model)
```
```text
def save_model(file, model, opt, with_opt=True, pickle_protocol=2):
    "Save `model` to `file` along with `opt` (if available, and if `with_opt`)"
    if rank_distrib(): return # don't save if child proc
    if opt is None: with_opt=False
    state = get_model(model).state_dict()
    if with_opt: state = {'model': state, 'opt':opt.state_dict()}
    torch.save(state, file, pickle_protocol=pickle_protocol)
```

-----


```python
print_source(rank_distrib)
```
```text
def rank_distrib():
    "Return the distributed rank of this process (if applicable)."
    return int(os.environ.get('RANK', 0))
```

-----


```python
print_source(get_model)
```
```text
def get_model(model):
    "Return the model maybe wrapped inside `model`."
    return model.module if isinstance(model, (DistributedDataParallel, nn.DataParallel)) else model
```

-----


```python
print_source(torch.save)
```
```text
def save(obj, f: Union[str, os.PathLike, BinaryIO, IO[bytes]],
         pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL, _use_new_zipfile_serialization=True) -> None:
    # Reference: https://github.com/pytorch/pytorch/issues/54354
    # The first line of this docstring overrides the one Sphinx generates for the
    # documentation. We need it so that Sphinx doesn't leak `pickle`s path from
    # the build environment (e.g. `<module 'pickle' from '/leaked/path').

    """save(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL, _use_new_zipfile_serialization=True)

    Saves an object to a disk file.

    See also: :ref:`saving-loading-tensors`

    Args:
        obj: saved object
        f: a file-like object (has to implement write and flush) or a string or
           os.PathLike object containing a file name
        pickle_module: module used for pickling metadata and objects
        pickle_protocol: can be specified to override the default protocol

    .. note::
        A common PyTorch convention is to save tensors using .pt file extension.

    .. note::
        PyTorch preserves storage sharing across serialization. See
        :ref:`preserve-storage-sharing` for more details.

    .. note::
        The 1.6 release of PyTorch switched ``torch.save`` to use a new
        zipfile-based file format. ``torch.load`` still retains the ability to
        load files in the old format. If for any reason you want ``torch.save``
        to use the old format, pass the kwarg ``_use_new_zipfile_serialization=False``.

    Example:
        >>> # Save to file
        >>> x = torch.tensor([0, 1, 2, 3, 4])
        >>> torch.save(x, 'tensor.pt')
        >>> # Save to io.BytesIO buffer
        >>> buffer = io.BytesIO()
        >>> torch.save(x, buffer)
    """
    _check_dill_version(pickle_module)

    with _open_file_like(f, 'wb') as opened_file:
        if _use_new_zipfile_serialization:
            with _open_zipfile_writer(opened_file) as opened_zipfile:
                _save(obj, opened_zipfile, pickle_module, pickle_protocol)
                return
        _legacy_save(obj, opened_file, pickle_module, pickle_protocol)
```

-----


```python
learn.save('1epoch')
```
```text
Path('/home/innom-dt/.fastai/data/imdb/models/1epoch.pth')
```

-----


```python
learn.load
```
```text
<bound method TextLearner.load of <fastai.text.learner.LMLearner object at 0x7fdb64fd4190>>
```

-----


```python
print_source(learn.load)
```
```text
    @delegates(load_model_text)
    def load(self, file, with_opt=None, device=None, **kwargs):
        if device is None: device = self.dls.device
        if self.opt is None: self.create_opt()
        file = join_path_file(file, self.path/self.model_dir, ext='.pth')
        load_model_text(file, self.model, self.opt, device=device, **kwargs)
        return self
```

-----


```python
load_model_text
```
```text
<function fastai.text.learner.load_model_text(file, model, opt, with_opt=None, device=None, strict=True)>
```

-----


```python
print_source(load_model_text)
```
```text
def load_model_text(file, model, opt, with_opt=None, device=None, strict=True):
    "Load `model` from `file` along with `opt` (if available, and if `with_opt`)"
    distrib_barrier()
    if isinstance(device, int): device = torch.device('cuda', device)
    elif device is None: device = 'cpu'
    state = torch.load(file, map_location=device)
    hasopt = set(state)=={'model', 'opt'}
    model_state = state['model'] if hasopt else state
    get_model(model).load_state_dict(clean_raw_keys(model_state), strict=strict)
    if hasopt and ifnone(with_opt,True):
        try: opt.load_state_dict(state['opt'])
        except:
            if with_opt: warn("Could not load the optimizer state.")
    elif with_opt: warn("Saved filed doesn't contain an optimizer state.")
```

-----


```python
distrib_barrier
```
```text
<function fastai.torch_core.distrib_barrier()>
```

-----


```python
print_source(distrib_barrier)
```
```text
def distrib_barrier():
    "Place a synchronization barrier in distributed training"
    if num_distrib() > 1 and torch.distributed.is_initialized(): torch.distributed.barrier()
```

-----


```python
learn = learn.load('1epoch')
```

-----


```python
learn.unfreeze()
learn.fit_one_cycle(10, 2e-3)
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>perplexity</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3.767039</td>
      <td>3.763731</td>
      <td>0.316231</td>
      <td>43.108986</td>
      <td>09:44</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3.692761</td>
      <td>3.705623</td>
      <td>0.323240</td>
      <td>40.675396</td>
      <td>09:29</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3.634718</td>
      <td>3.654817</td>
      <td>0.328937</td>
      <td>38.660458</td>
      <td>09:31</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3.563724</td>
      <td>3.624163</td>
      <td>0.332917</td>
      <td>37.493317</td>
      <td>09:32</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3.486968</td>
      <td>3.600153</td>
      <td>0.335374</td>
      <td>36.603825</td>
      <td>09:36</td>
    </tr>
    <tr>
      <td>5</td>
      <td>3.435516</td>
      <td>3.585277</td>
      <td>0.337806</td>
      <td>36.063351</td>
      <td>09:30</td>
    </tr>
    <tr>
      <td>6</td>
      <td>3.363010</td>
      <td>3.575442</td>
      <td>0.339413</td>
      <td>35.710388</td>
      <td>09:18</td>
    </tr>
    <tr>
      <td>7</td>
      <td>3.300442</td>
      <td>3.574242</td>
      <td>0.340387</td>
      <td>35.667561</td>
      <td>09:22</td>
    </tr>
    <tr>
      <td>8</td>
      <td>3.247055</td>
      <td>3.576924</td>
      <td>0.340627</td>
      <td>35.763359</td>
      <td>09:14</td>
    </tr>
    <tr>
      <td>9</td>
      <td>3.210976</td>
      <td>3.581657</td>
      <td>0.340366</td>
      <td>35.933022</td>
      <td>09:18</td>
    </tr>
  </tbody>
</table>
</div>


#### Encoder
* the model not including the task-specific final layer(s)
* typically used to refer to the body of NLP and generative models

-----


```python
learn.save_encoder
```
```text
<bound method TextLearner.save_encoder of <fastai.text.learner.LMLearner object at 0x7fdb64fd4190>>
```

-----


```python
print_source(learn.save_encoder)
```
```text
    def save_encoder(self, file):
        "Save the encoder to `file` in the model directory"
        if rank_distrib(): return # don't save if child proc
        encoder = get_model(self.model)[0]
        if hasattr(encoder, 'module'): encoder = encoder.module
        torch.save(encoder.state_dict(), join_path_file(file, self.path/self.model_dir, ext='.pth'))
```

-----


```python
learn.save_encoder('finetuned')
```
### Text Generation
* Training the model to predict the next word of a sentence enables it to generate new reviews

-----


```python
# Prompt
TEXT = "I liked this movie because"
# Generate 40 new words
N_WORDS = 40
N_SENTENCES = 2
# Add some randomness (temperature) to prevent generating the same review twice 
preds = [learn.predict(TEXT, N_WORDS, temperature=0.75) 
         for _ in range(N_SENTENCES)]
```

-----

```python
print("\n".join(preds))
```
```text
i liked this movie because it was a very well done film . Lee Bowman does a wonderful job in his role . Being a big Astaire fan , i think this movie is worth seeing for the musical numbers .
i liked this movie because it was based on a true story . The script was excellent as it was great . i would recommend this movie to anyone interested in history and the history of the holocaust . It was great to
```


**Note:** The model has learned a lot about English sentences, despite not having any explicitely programmed knowledge.

### Creating the Classifier DataLoaders
* very similar to the DataBlocks used for the image classification datasets
* data augmentation has not been well-explored
* need to pad smaller documents when creating mini-batches
    * batches are padded based on the largest document in a given batch
* the data block API automatically handles sorting and padding when using TextBlock with `is_lm=False`

-----


```python
GrandparentSplitter
```
```text
<function fastai.data.transforms.GrandparentSplitter(train_name='train', valid_name='valid')>
```

-----


```python
print_source(GrandparentSplitter)
```
```text
def GrandparentSplitter(train_name='train', valid_name='valid'):
    "Split `items` from the grand parent folder names (`train_name` and `valid_name`)."
    def _inner(o):
        return _grandparent_idxs(o, train_name),_grandparent_idxs(o, valid_name)
    return _inner
```

-----


```python
dls_clas = DataBlock(
    # Pass in the vocab used by the language model
    blocks=(TextBlock.from_folder(path, vocab=dls_lm.vocab),CategoryBlock),
    get_y = parent_label,
    get_items=partial(get_text_files, folders=['train', 'test']),
    splitter=GrandparentSplitter(valid_name='test')
).dataloaders(path, path=path, bs=128, seq_len=72)
```

-----


```python
dls_clas.show_batch(max_n=3)
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>xxbos xxmaj match 1 : xxmaj tag xxmaj team xxmaj table xxmaj match xxmaj bubba xxmaj ray and xxmaj spike xxmaj dudley vs xxmaj eddie xxmaj guerrero and xxmaj chris xxmaj benoit xxmaj bubba xxmaj ray and xxmaj spike xxmaj dudley started things off with a xxmaj tag xxmaj team xxmaj table xxmaj match against xxmaj eddie xxmaj guerrero and xxmaj chris xxmaj benoit . xxmaj according to the rules of the match , both opponents have to go through tables in order to get the win . xxmaj benoit and xxmaj guerrero heated up early on by taking turns hammering first xxmaj spike and then xxmaj bubba xxmaj ray . a xxmaj german xxunk by xxmaj benoit to xxmaj bubba took the wind out of the xxmaj dudley brother . xxmaj spike tried to help his brother , but the referee restrained him while xxmaj benoit and xxmaj guerrero</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>xxbos xxmaj by now you 've probably heard a bit about the new xxmaj disney dub of xxmaj miyazaki 's classic film , xxmaj laputa : xxmaj castle xxmaj in xxmaj the xxmaj sky . xxmaj during late summer of 1998 , xxmaj disney released " kiki 's xxmaj delivery xxmaj service " on video which included a preview of the xxmaj laputa dub saying it was due out in " 1 xxrep 3 9 " . xxmaj it 's obviously way past that year now , but the dub has been finally completed . xxmaj and it 's not " laputa : xxmaj castle xxmaj in xxmaj the xxmaj sky " , just " castle xxmaj in xxmaj the xxmaj sky " for the dub , since xxmaj laputa is not such a nice word in xxmaj spanish ( even though they use the word xxmaj laputa many times</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>xxbos xxmaj heavy - handed moralism . xxmaj writers using characters as mouthpieces to speak for themselves . xxmaj predictable , plodding plot points ( say that five times fast ) . a child 's imitation of xxmaj britney xxmaj spears . xxmaj this film has all the earmarks of a xxmaj lifetime xxmaj special reject . \n\n i honestly believe that xxmaj jesus xxmaj xxunk and xxmaj julia xxmaj xxunk set out to create a thought - provoking , emotional film on a tough subject , exploring the idea that things are not always black and white , that one who is a criminal by definition is not necessarily a bad human being , and that there can be extenuating circumstances , especially when one puts the well - being of a child first . xxmaj however , their earnestness ends up being channeled into preachy dialogue and trite</td>
      <td>neg</td>
    </tr>
  </tbody>
</table>
</div>
-----

```python
nums_samp = toks200[:10].map(num)
```

-----


```python
nums_samp.map(len)
```
```text
(#10) [177,562,211,125,125,425,421,1330,196,278]
```

-----


```python
learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, 
                                metrics=accuracy).to_fp16()
```


```python
learn = learn.load_encoder('finetuned')
```

### Fine-Tuning the Classifier
* NLP classifiers benefit from gradually unfreezing a few layers at a time

-----


```python
learn.fit_one_cycle(1, 2e-2)
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.242196</td>
      <td>0.178359</td>
      <td>0.931280</td>
      <td>00:29</td>
    </tr>
  </tbody>
</table>
</div>
-----

```python
# Freeze all except the last two parameter groups
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2))
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.226292</td>
      <td>0.162955</td>
      <td>0.938840</td>
      <td>00:35</td>
    </tr>
  </tbody>
</table>
</div>
-----

```python
# Freeze all except the last two parameter groups
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.150115</td>
      <td>0.144669</td>
      <td>0.947280</td>
      <td>00:46</td>
    </tr>
  </tbody>
</table>
</div>
-----

```python
# Unfreeze all layers
learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3))
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.160042</td>
      <td>0.149997</td>
      <td>0.945320</td>
      <td>00:54</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.146106</td>
      <td>0.148102</td>
      <td>0.945320</td>
      <td>00:54</td>
    </tr>
  </tbody>
</table>
</div>


**Note:** We can further improve the accuracy by training another model on all the texts read backward and averaging the predictions of the two models.



## Disinformation and Language Models
* Even simple algorithms based on rules could be used to create fraudulent accounts and try influence policymakers
* [More than a Million Pro-Repeal Net Neutrality Comments were Likely Faked](https://hackernoon.com/more-than-a-million-pro-repeal-net-neutrality-comments-were-likely-faked-e9f0e3ed36a6)
    * Jeff Kao discovered a large cluster of comments opposing net neutrality that seemed to have been generated by some sort of Mad Libs-style mail merge.
    * estimated that less than 800,000 of the 22M+ comments could be considered unique
    * more than 99% of the truly unique comments were in favor of net neutrality
* The same type of language model as trained above could be used to generate context-appropriate, believable text






## References

* [Deep Learning for Coders with fastai & PyTorch](https://www.oreilly.com/library/view/deep-learning-for/9781492045519/)
* [The fastai book GitHub Repository](https://github.com/fastai/fastbook)



**Previous:** [Notes on fastai Book Ch. 9](../chapter-9/)

**Next:** [Notes on fastai Book Ch. 11](../chapter-11/)







<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->