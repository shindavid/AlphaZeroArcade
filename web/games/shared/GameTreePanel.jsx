import React, { useMemo, useCallback } from 'react';
import ReactFlow, { Background, Controls } from 'reactflow';
import 'reactflow/dist/style.css';
import './GameTree.css'

import { getLayoutElements } from './GameTreeUtils';
import { GameTreeNode } from './GameTreeNode';

const nodeTypes = {
  gameNode: GameTreeNode,
};

export function GameTreePanel({ history, seatToHtml, onBacktrack }) {

  const { nodes, edges } = useMemo(() => {
    return getLayoutElements(history, seatToHtml);
  }, [history, seatToHtml]);

  const onNodeClick = useCallback((event, node) => {
    if (onBacktrack && node.data && typeof node.data.index !== 'undefined') {
      console.log('Node clicked:', node.data.index);
      onBacktrack(node.data.index);
    }
  }, [onBacktrack]);

  return (
    <div className="game-tree-container">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        nodesDraggable={false}
        onNodeClick={onNodeClick}
      >
        <Background className="game-tree-background" gap={16} />
        <Controls />
      </ReactFlow>
    </div>
  );
}
