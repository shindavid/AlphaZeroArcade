# Shared Code and Assets

This directory contains code, assets, and styles shared between all games (e.g., tictactoe, connect4).

- Place reusable React/Vue/Svelte components in `components/`
- Place JS/TS utility functions in `utils/`
- Place images and static assets in `assets/`
- Place CSS/SCSS modules in `styles/`

## Usage

Import shared modules in your game code, e.g.:
```js
import Logo from '../shared/assets/vite.svg';
import { someUtil } from '../shared/utils/someUtil';
```

## Conventions
- Only put code/assets here if they are truly shared or likely to be reused.
- If you refactor something from a game into here, update all import paths accordingly.
- Document any non-obvious shared code in this README.
