import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Box from '@mui/material/Box';
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
        <Box sx={{ flexGrow: 1 }}>
          <Logo
            variant="h6"
            component={RouterLink}
            to="/"
          >
            AI-NLP Interview Assistant
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
