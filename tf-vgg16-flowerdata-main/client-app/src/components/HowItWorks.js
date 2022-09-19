const pStyles = {
  margin: "2rem",
};
export default function HowItWorks() {
  return (
    <div className="layout-container">
      <h1>¿Cómo funciona?</h1>
      <p style={pStyles}>
        Esta aplicación utiliza inteligencia articial. Si no estás familiarizado
        con este concepto, no te preocupes. En términos muy simples una
        inteligencia artificial es un programa de computadora que tiene la
        capacidad de aprender ciertas cosas.
      </p>
      <p style={pStyles}>
        Por ejemplo, una persona puede memorizar algo nuevo si estudia lo
        suficiente. Si una persona todos los días aprende de plantas, cómo lucen
        y cómo se llaman, poco a poco será capaz de reconocerlas.
      </p>

      <p style={pStyles}>
        Con las computadoras es similar, la ventaja es que estas son muy rápidas
        y pueden analizar información muchísimo más rápido que una persona. Lo
        que hicimos en este proyecto fue asignarle a un programa la tarea ver
        miles de imágenes de cinco categorías de plantas, esto hasta que se
        hiciera bueno en reconocerlas.
      </p>

      <p style={pStyles}>
        El proceso de entrenamiento puede ser muy lento y tedioso, puede
        conllevar horas o incluso días. Sin embargo, una vez entrenado se le
        puede preguntar cosas y responde muy rápido.
      </p>
    </div>
  );
}
