import Circle from "./Circle";


export default function Column({ column, onColumnClick}) {
    return (
      <div className="column">
        <Circle player={column[0]} onCircleClick={onColumnClick} />
        <Circle player={column[1]} onCircleClick={onColumnClick} />
        <Circle player={column[2]} onCircleClick={onColumnClick} />
        <Circle player={column[3]} onCircleClick={onColumnClick} />
        <Circle player={column[4]} onCircleClick={onColumnClick} />
        <Circle player={column[5]} onCircleClick={onColumnClick} />
      </div>
    );
}