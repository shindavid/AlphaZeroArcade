This directory contains third-party repositories added via `git subtree`.

The tinyexpr directory was added via this command:

```
git subtree add --prefix=extra_deps/tinyexpr https://github.com/codeplea/tinyexpr.git master --squash
```

To update to the latest from the source repository, run:

```
git subtree pull --prefix=extra_deps/tinyexpr https://github.com/codeplea/tinyexpr.git master --squash
```

and then push the changes.

Similarly, we have:

```
EigenRand: https://github.com/bab2min/EigenRand.git (main)
connect4: https://github.com/PascalPons/connect4.git (master)
edax-reversi: https://github.com/abulmo/edax-reversi.git (master)
```

Additionally, the file 7x6.book was cloned from:

```
https://github.com/PascalPons/connect4/releases/download/book/7x6.book
```

