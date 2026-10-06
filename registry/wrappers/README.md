# Model wrappers

Small torch modules that a card's `builder` points at when the published code
does preprocessing or output handling the input contract cannot express
(checkpoint-stored normalisation statistics, padding after normalisation, a
dict-returning forward). Each file has a distinct module name because card
`code_path` entries share one `sys.path` (L2-6). Cards point at the server copy in
`/data/mi-val/models/_wrappers`, kept apart from any checkout so a running
study never sees a wrapper change under it; copy a file there when it changes.
