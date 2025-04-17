import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box'; // Import Box
import logo from '../assets/logo.png'; // Import the logo
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
// Removed duplicate Box import
import { styled } from '@mui/material/styles';

// Custom styled component for the logo
const Logo = styled(Typography)(({ theme }) => ({
  fontWeight: 700,
  color: 'white',
  textDecoration: 'none',
}));

// Custom styled component for the navigation buttons
const NavButton = styled(Button)(({ theme }) => ({
  color: 'white',
  marginLeft: theme.spacing(2),
}));

const Header = () => {
  return (
    <AppBar position="static">
      <Toolbar>
        {/* Replace text logo with image and text */}
        <Box sx={{ flexGrow: 1, display: 'flex', alignItems: 'center' }}>
          <img src={logo} alt="PrepIQ Logo" style={{ height: '40px', marginRight: '10px' }} />
          <Logo
            variant="h6"
            component={RouterLink}
            to="/"
          >
            PrepIQ
          </Logo>
        </Box>
        <NavButton color="inherit" component={RouterLink} to="/">
          Home
        </NavButton>
        <NavButton color="inherit" component={RouterLink} to="/interview">
          Start Interview
        </NavButton>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
