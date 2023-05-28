const playerColor = {
    0: 'red',
    1: 'gold',
    null: 'white',
};


export default function Circle({ player, onCircleClick }) {
    return (
        <button 
            className="circle"
            onClick={onCircleClick}
            style={{ background: playerColor[player] }}
        ></button>
    );
}