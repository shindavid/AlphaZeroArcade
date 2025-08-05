// Example shared React component
import React from 'react';
import '../styles/shared.css';

export function SharedButton({ children, ...props }) {
  return (
    <button className="shared-button" {...props}>
      {children}
    </button>
  );
}
