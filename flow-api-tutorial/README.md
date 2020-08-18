It resolves the issue: [#134](https://github.com/jina-ai/examples/issues/134)

# Overview
There are 3 ways to use index and search method in Jina, we have examples like faiss, southpark and flower-search. This note will help you understand the pattern beyond those 3 examples.

#3 ways to initiate index flow
<p>
 
1.index_ndarray supports ndarray, and you can use this method to index your array data, like pics and sounds. Here is the example:[Faiss Search](https://github.com/jina-ai/examples/tree/master/faiss-search)

```python
data_path = os.path.join('[Your data file]')
flow = Flow().load_config('flow-index.yml')
with flow.build() as fl:
    fl.index_ndarray(read_data(data_path), batch_size=batch_size)
```
2.index_files supports you to load files. In our example we use jpg files.Here is the example:[flower Search](https://github.com/jina-ai/examples/tree/master/flower-search)
```python
f = Flow().load_config('flow-index.yml')
with f:
    f.index_files(f'{data_path}/*.jpg', size=num_docs, read_mode='rb', batch_size=2)
```
3.index_lines supports you to load lines, which you can only use it in text.Here is the example: [Southpark Search](https://github.com/jina-ai/examples/tree/master/southpark-search)
```python
config(num_docs, mode = 'index')
data_path = os.path.join(os.environ['DATA_DIR'], os.environ['DATA_FILE'])
f = Flow().load_config('flow-index.yml')
with f:
    f.index_lines(filepath=data_path, batch_size=8, size=int(os.environ['MAX_NUM_DOCS']))
```

#3 ways to initiate query flow
<p>

1.search_ndarray supports you to search ndarray by ANN models. Here is the example[Faiss Search](https://github.com/jina-ai/examples/tree/master/faiss-search)

```python
data_path = os.path.join('[Your data file]')
flow = Flow().load_config('flow-query.yml')
with flow.build() as fl:
   ppr = lambda x: save_topk(x, os.path.join(os.environ['TMP_DATA_DIR'], 'query_results.txt'), top_k)
   fl.search_ndarray(read_data(data_path), output_fn=ppr, top_k=top_k)
```

2.search_files helps you to search from files.Here is the example:[flower Search](https://github.com/jina-ai/examples/tree/master/flower-search)
```python
f = Flow().load_config('flow-query.yml')
f.search_files(f'{data_path}/*.jpg', size=num_docs, read_mode='rb', batch_size=2)
with f:
    f.block()
```
3.search_lines helps you to search in lines.Here is the example: [Southpark Search](https://github.com/jina-ai/examples/tree/master/southpark-search)
```python
config(num_docs, mode = 'search')
f = Flow().load_config('flow-query.yml')
with f:
    text = input('please type a sentence: ')
    ppr = lambda x: print_topk(x, text)
    f.search_lines(lines=[text, ], output_fn=ppr, top_k=top_k)
```
