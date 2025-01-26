import { useState } from 'react';
import axios from 'axios';
import { WiDaySunny } from 'react-icons/wi';

const API_KEY = '048313ac996327915f2e6e423a304b09';

export default function UVIndex() {
  const [location, setLocation] = useState('');
  const [weatherData, setWeatherData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      // First get coordinates for the location
      const geoResponse = await axios.get(
        `https://api.openweathermap.org/geo/1.0/direct?q=${location}&limit=1&appid=${API_KEY}`
      );

      if (geoResponse.data.length === 0) {
        throw new Error('Location not found');
      }

      const { lat, lon } = geoResponse.data[0];

      // Get weather data including UV index
      const weatherResponse = await axios.get(
        `https://api.openweathermap.org/data/2.5/onecall?lat=${lat}&lon=${lon}&exclude=minutely,hourly&appid=${API_KEY}&units=metric`
      );

      setWeatherData(weatherResponse.data);
    } catch (err) {
      setError('Error fetching weather data. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const getUVAdvice = (uvIndex) => {
    if (uvIndex >= 11) {
      return {
        risk: 'Extreme',
        color: '#6c1577',
        advice: 'Take all precautions: Avoid sun exposure during midday hours. Shirt, sunscreen, and hat are essential.'
      };
    } else if (uvIndex >= 8) {
      return {
        risk: 'Very High',
        color: '#d8001d',
        advice: 'Extra precautions needed. Minimize sun exposure during midday hours. Apply SPF 50+ sunscreen.'
      };
    } else if (uvIndex >= 6) {
      return {
        risk: 'High',
        color: '#f95901',
        advice: 'Protection required. Reduce time in the sun between 10 a.m. and 4 p.m. Apply SPF 30+ sunscreen.'
      };
    } else if (uvIndex >= 3) {
      return {
        risk: 'Moderate',
        color: '#f7e401',
        advice: 'Stay in shade near midday. Wear sunscreen and protective clothing.'
      };
    } else {
      return {
        risk: 'Low',
        color: '#299501',
        advice: 'No protection required. Safe to be outside.'
      };
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-pink-50 py-12 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-2xl shadow-xl p-8 border border-pink-100">
          <div className="text-center mb-8">
            <WiDaySunny className="text-6xl text-yellow-400 mx-auto mb-4" />
            <h1 className="text-3xl font-bold text-gray-800">Weather & UV Index</h1>
            <p className="text-gray-600 mt-2">Get weather information and skin care recommendations</p>
          </div>

          <form onSubmit={handleSubmit} className="mb-8">
            <div className="flex flex-col md:flex-row gap-4">
              <div className="flex-1 relative">
                <input
                  type="text"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  placeholder="Enter city name..."
                  className="w-full px-4 py-3 rounded-lg border border-pink-200 focus:outline-none focus:ring-2 focus:ring-pink-500/50"
                />
              </div>
              <button
                type="submit"
                className="px-6 py-3 bg-pink-500 text-white rounded-lg hover:bg-pink-600 transition-all duration-200 disabled:opacity-50"
                disabled={loading}
              >
                {loading ? 'Loading...' : 'Check Weather'}
              </button>
            </div>
          </form>

          {error && (
            <div className="text-center py-4 text-red-500">
              {error}
            </div>
          )}

          {weatherData && (
            <div className="bg-gradient-to-r from-pink-50 to-purple-50 rounded-xl p-6">
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <h2 className="text-2xl font-semibold mb-4">Current Weather</h2>
                  <div className="space-y-4 bg-white/50 rounded-lg p-4">
                    <p className="text-lg">
                      Temperature: <span className="text-pink-600">{weatherData.current.temp}°C</span>
                    </p>
                    <p className="text-lg">
                      Weather: <span className="text-pink-600">{weatherData.current.weather[0].main}</span>
                    </p>
                    <p className="text-lg">
                      UV Index: 
                      <span 
                        className={`ml-2 px-3 py-1 rounded-full text-white ${getUVAdvice(weatherData.current.uvi).color}`}
                      >
                        {weatherData.current.uvi} - {getUVAdvice(weatherData.current.uvi).risk}
                      </span>
                    </p>
                  </div>
                </div>

                <div>
                  <h2 className="text-2xl font-semibold mb-4">Skin Care Recommendations</h2>
                  <div className="bg-white/50 rounded-lg p-4">
                    <p className="text-lg leading-relaxed">
                      {getUVAdvice(weatherData.current.uvi).advice}
                    </p>
                    <div className="mt-4 p-4 bg-white/50 rounded-lg">
                      <h3 className="font-semibold mb-2">Additional Precautions:</h3>
                      <ul className="list-disc list-inside space-y-2">
                        <li>Use broad-spectrum sunscreen</li>
                        <li>Wear protective clothing</li>
                        <li>Use sunglasses with UV protection</li>
                        <li>Stay hydrated</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 