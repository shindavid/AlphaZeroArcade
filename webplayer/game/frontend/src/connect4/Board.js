import Column from "./Column";


export default function Board({ redIsNext, columns, onPlay }) {
    function handleClick(colIdx) {
        let nextColumns = columns.slice()
        let rowIdx = nextColumns[colIdx].lastIndexOf(null);
        nextColumns[colIdx][rowIdx] = redIsNext ? 0 : 1;
        onPlay(nextColumns)
    }

    return (
        <div className="board">
            <Column column={columns[0]} onColumnClick={() => handleClick(0)} />
            <Column column={columns[1]} onColumnClick={() => handleClick(1)} />
            <Column column={columns[2]} onColumnClick={() => handleClick(2)} />
            <Column column={columns[3]} onColumnClick={() => handleClick(3)} />
            <Column column={columns[4]} onColumnClick={() => handleClick(4)} />
            <Column column={columns[5]} onColumnClick={() => handleClick(5)} />
            <Column column={columns[6]} onColumnClick={() => handleClick(6)} />
        </div>
    );
}
