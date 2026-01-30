// Quick script to check if backend is running
const http = require('http');

const options = {
  hostname: 'localhost',
  port: 5000,
  path: '/api/health',
  method: 'GET'
};

const req = http.request(options, (res) => {
  console.log(`✅ Backend is running! Status: ${res.statusCode}`);
  console.log(`   URL: http://localhost:5000`);
  
  let data = '';
  res.on('data', (chunk) => {
    data += chunk;
  });
  
  res.on('end', () => {
    try {
      const json = JSON.parse(data);
      console.log(`   Response:`, json);
    } catch (e) {
      console.log(`   Response:`, data);
    }
  });
});

req.on('error', (error) => {
  console.log(`❌ Backend is NOT running!`);
  console.log(`   Error: ${error.message}`);
  console.log(`\n   To start the backend, run:`);
  console.log(`   cd backend && npm start`);
});

req.end();
