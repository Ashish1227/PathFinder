import React, { useState } from 'react';

function App() {
  const [formData, setFormData] = useState({
    tags: '',
    difficulty: '',
    averageRating: 0,
    length: 0,
  });
  const [recommendations, setRecommendations] = useState([]);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setFormData((prevData) => ({ ...prevData, [name]: value }));
  };

  const handleSubmit = (event) => {
    event.preventDefault();

    fetch('http://127.0.0.1:5000/recommend_tfidf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData),
    })
      .then(response => response.json())
      .then(data => {
        console.log('Recommendations:', data);
        setRecommendations(data.recommendations);  // Update state with recommendations
      })
      .catch(error => console.error('Error fetching recommendations:', error));
  };

  return (
    <div className="App">
      <h1>Trek Recommendation System</h1>
      <form onSubmit={handleSubmit}>
        <label htmlFor="tags">Tags (e.g., hiking, scenic):</label>
        <input type="text" id="tags" name="tags" value={formData.tags} onChange={handleChange} />
        <label htmlFor="difficulty">Difficulty:</label>
        <input type="text" id="difficulty" name="difficulty" value={formData.difficulty} onChange={handleChange} />
        <label htmlFor="averageRating">Average Rating (0-5):</label>
        <input type="number" min="0" max="5" id="averageRating" name="averageRating" value={formData.averageRating} onChange={handleChange} />
        <label htmlFor="length">Length (km):</label>
        <input type="number" min="0" id="length" name="length" value={formData.length} onChange={handleChange} />
        <button type="submit">Find Recommendations</button>
      </form>

      {/* Display recommendations in a table if recommendations state is not empty */}
      {recommendations.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Trail Name</th>
              <th>Similarity Score</th>
            </tr>
          </thead>
          <tbody>
            {recommendations.map((recommendation, index) => (
              <tr key={index}>
                <td>{recommendation[0]}</td>  {/* Access trail name from the first element */}
                <td>{recommendation[1].toFixed(4)}</td>  {/* Format score with 4 decimal places */}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default App;

