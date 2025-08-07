export class Connect4Animation {
  constructor() {
    this.timer = null;
    this.state = null; // { col, row, disc, targetRow, animRow, onComplete }
  }

  start({ col, row, disc, onComplete, interval = 60, onFrame }) {
    this.end();
    this.state = {
      col,
      row,
      disc,
      targetRow: row,
      animRow: 0,
      onComplete,
    };
    let animationDone = false;
    this.timer = setInterval(() => {
      if (this.state.animRow < row) {
        this.state.animRow += 1;
        if (onFrame) onFrame(this.get());
      } else {
        if (!animationDone) {
          animationDone = true;
          this.end();
          if (onComplete) onComplete();
        }
      }
    }, interval);
    if (onFrame) onFrame(this.get());
  }

  end() {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    this.state = null;
  }

  isActive() {
    return !!this.state;
  }

  get() {
    return this.state;
  }
}
