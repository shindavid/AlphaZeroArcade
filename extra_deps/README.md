This directory contains third-party repositories added via `git subtree`.

The EigenRand directory was added via this command:

```
git subtree add --prefix=extra_deps/EigenRand https://github.com/bab2min/EigenRand.git main --squash
```

To update to the latest from the source repository, run:

```
git subtree pull --prefix=extra_deps/EigenRand https://github.com/bab2min/EigenRand.git main --squash
```

and then push the changes.

Similarly, we have:

```
connect4: https://github.com/PascalPons/connect4.git (master)
edax-reversi: https://github.com/abulmo/edax-reversi.git (master)
```

Additionally, the connect4 7x6.book file was cloned from:

```
https://github.com/PascalPons/connect4/releases/download/book/7x6.book
```

And the edax-reversi data/ dir was cloned from:

```
https://github.com/abulmo/edax-reversi/releases/download/v4.4/eval.7z
```

