import React from "react";
import { Tab, Tabs, TabList, TabPanel } from "react-tabs";
import Home from "./Home";
import "../Tabs.css";
import About from "../components/About";
import HowItWorks from "./HowItWorks";

function HomeTabs() {
  return (
    <div>
      <Tabs>
        <TabList>
          <Tab>Inicio</Tab>
          <Tab>¿Cómo funciona?</Tab>
          <Tab>Sobre el proyecto</Tab>
        </TabList>
        <TabPanel style={{marginTop:'3rem'}}>
          <Home />
        </TabPanel>
        <TabPanel style={{marginTop:'3rem'}}>
          <HowItWorks />
        </TabPanel>
        <TabPanel style={{marginTop:'3rem'}}>
          <About />
        </TabPanel>
      </Tabs>
    </div>
  );
}

export default HomeTabs;
