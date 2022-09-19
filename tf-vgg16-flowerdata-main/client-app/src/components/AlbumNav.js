import React, { Component } from "react";
import { Route, NavLink, HashRouter } from "react-router-dom";
import Album1 from "./Album1";
import Album2 from "./Album2";
import Album3 from "./Album3";

class AlbumNav extends Component {
  render() {
    return (
      <HashRouter>
        <div>
          <h1>List of Albums</h1>
          <ul className="header">
            <li>
              <NavLink exact to="/Album1">
                My Adventures
              </NavLink>
            </li>
            <li>
              <NavLink to="/Album2">Food Trips</NavLink>
            </li>
            <li>
              <NavLink to="/Album3">My Family</NavLink>
            </li>
          </ul>
          <div className="content">
            <Route exact path="/Album1" component={Album1} />
            <Route path="/Album2" component={Album2} />
            <Route path="/Album3" component={Album3} />
          </div>
        </div>
      </HashRouter>
    );
  }
}

export default AlbumNav;
