// frontend/src/pages/BacktestPage.tsx
import React, { useState } from 'react';

const BacktestPage = () => {
  const [error, setError] = useState('');
  const [results, setResults] = useState([]);

  // ... other component logic

  return (
    <div>
      {/* ... other JSX */}

      {error && !results.length && (
        <p style={{ color: 'red' }}>{error}</p>
      )}

      {!error && results.length > 0 && (
        <table>
          {/* table contents */}
        </table>
      )}

      {/* ... other JSX */}
    </div>
  );
};

export default BacktestPage;