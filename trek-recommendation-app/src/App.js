import React, { useState } from 'react';

function App() {
  const [formData, setFormData] = useState({
    tags: '',
    difficulty: '',
    est_time: 0,
    length: 0,
  });
  const [recommendations, setRecommendations] = useState([]);
  const [selectedRecommendation, setSelectedRecommendation] = useState(null);
  const [step, setStep] = useState(1);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setFormData((prevData) => ({ ...prevData, [name]: value }));
  };

  const handleSelectRecommendation = (recommendation) => {
    setSelectedRecommendation(recommendation);
  };

  const handleRatingSubmit = (event) => {
    event.preventDefault();
  
    const rating = event.target.elements.rating.value; // Get rating from form
  
    fetch('http://127.0.0.1:5000/rate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        rating: rating
      }),
    })
      .then(response => response.json())
      .then(data => {
        console.log('Rating submitted:', data);
        setStep(1);
        setFormData({
          tags: '',
          difficulty: '',
          est_time: 0,
          length: 0,
        });
      })
      .catch(error => console.error('Error submitting rating:', error));
  };

  const handleSubmit = (event) => {
    event.preventDefault();

    fetch('http://127.0.0.1:5000/recommend_cucb', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData),
    })
      .then(response => response.json())
      .then(data => {
        console.log('Recommendations:', data);
        setRecommendations(data.recommendations); 
        setStep(step+1);
      })
      .catch(error => console.error('Error fetching recommendations:', error));
  };

  const handleNext = () => {
    setStep(step + 1); // Move to next step
  };
  return (
    <div className="App">
      <h1>Trek Recommendation System</h1>
      {step === 1 && (
        <form onSubmit={handleSubmit}>
          <label htmlFor="tags">Tags (e.g., hiking, scenic):</label>
          <input type="text" id="tags" name="tags" value={formData.tags} onChange={handleChange} />
          <label htmlFor="difficulty">Difficulty:</label>
          <input type="text" id="difficulty" name="difficulty" value={formData.difficulty} onChange={handleChange} />
          <label htmlFor="est_time">Estimated Time:</label>
          <input type="number" min="0" id="est_time" name="est_time" value={formData.est_time} onChange={handleChange} />
          <label htmlFor="length">Length (km):</label>
          <input type="number" min="0" id="length" name="length" value={formData.length} onChange={handleChange} />
          <button type="submit">Find Recommendations</button>
        </form>
      )}
      {step === 2 && recommendations.length > 0 && (
        <div>
          <h2>Recommended Trek:</h2>
          <p>{recommendations}</p>  {/* Access trail name from the first element */}
          <button type="button" onClick={handleNext}>Next</button>
        </div>
      )}
      {step === 3 && (
        <div>
          <h2>Rate Your Experience:</h2>
          <form onSubmit={handleRatingSubmit}>
            <label htmlFor="rating">Rating (1-5):</label>
            <input type="number" min="1" max="5" id="rating" name="rating" required />
            <button type="submit">Submit Rating</button>
          </form>
        </div>
      )}
    </div>
  );
}

export default App;

