import React, { useState } from 'react';

const PokemonCardEtiquetteGenerator = () => {
  // Couleurs préétablies pour les différents types de cartes Pokémon
  const pokemonTypes = {
    "Feu": {
      primary: "#FF4422",
      secondary: "#FFCC33",
      text: "#FFFFFF",
      background: "#FFEECC"
    },
    "Eau": {
      primary: "#3399FF",
      secondary: "#66CCFF",
      text: "#FFFFFF",
      background: "#E6F5FF"
    },
    "Plante": {
      primary: "#77CC55",
      secondary: "#AADD88",
      text: "#FFFFFF",
      background: "#EEFFEE"
    },
    "Électrique": {
      primary: "#FFDD33",
      secondary: "#FFFF66",
      text: "#333333",
      background: "#FFFFDD"
    },
    "Psy": {
      primary: "#FF5599",
      secondary: "#FF99CC",
      text: "#FFFFFF",
      background: "#FFEEF5"
    },
    "Combat": {
      primary: "#BB5544",
      secondary: "#DD8866",
      text: "#FFFFFF",
      background: "#F5E6E6"
    },
    "Ténèbres": {
      primary: "#775544",
      secondary: "#AA8866",
      text: "#FFFFFF",
      background: "#EEEAE6"
    },
    "Dragon": {
      primary: "#7766EE",
      secondary: "#AACCFF",
      text: "#FFFFFF",
      background: "#EEEEFF"
    },
    "Normal": {
      primary: "#AAAAAA",
      secondary: "#DDDDDD",
      text: "#333333",
      background: "#F5F5F5"
    }
  };

  const [formData, setFormData] = useState({
    nomCarte: 'DRACAUFEU',
    edition: 'ED.1',
    serie: 'NEO REVELATION - 66',
    annee: '2002 - WOTC - FR',
    numeroSerie: '123456789',
    noteGlobale: '9.5',
    noteCentrage: '9.5',
    noteCoins: '10',
    noteCotes: '9',
    noteSurface: '9.5',
    etatLabel: 'NEUF+',
    logoText: 'CCC GRADING',
    selectedType: 'Feu',
    logoOption: 'default' // 'default', 'custom', 'text', 'upload'
  });

  const [customColors, setCustomColors] = useState({
    primary: "#FF4422",
    secondary: "#FFCC33"
  });
  
  const [uploadedLogo, setUploadedLogo] = useState(null);
  const [logoPreview, setLogoPreview] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevState => ({
      ...prevState,
      [name]: value
    }));

    // Si le type de Pokémon change, mettre à jour les couleurs
    if (name === 'selectedType') {
      setCustomColors({
        primary: pokemonTypes[value].primary,
        secondary: pokemonTypes[value].secondary
      });
    }
  };

  const handleColorChange = (e) => {
    const { name, value } = e.target;
    setCustomColors(prevState => ({
      ...prevState,
      [name]: value
    }));
  };

  // Styles en ligne pour l'interface
  const styles = {
    container: {
      fontFamily: 'Arial, sans-serif',
      maxWidth: '800px',
      margin: '0 auto',
      padding: '20px'
    },
    formSection: {
      marginBottom: '20px',
      padding: '15px',
      border: '1px solid #ddd',
      borderRadius: '5px',
      backgroundColor: pokemonTypes[formData.selectedType].background
    },
    inputGroup: {
      display: 'flex',
      flexWrap: 'wrap',
      gap: '10px',
      marginBottom: '15px'
    },
    inputField: {
      display: 'flex',
      flexDirection: 'column',
      width: 'calc(50% - 10px)',
      minWidth: '200px'
    },
    label: {
      fontWeight: 'bold',
      marginBottom: '5px',
      fontSize: '14px'
    },
    input: {
      padding: '8px',
      border: '1px solid #ccc',
      borderRadius: '4px'
    },
    select: {
      padding: '8px',
      border: '1px solid #ccc',
      borderRadius: '4px',
      backgroundColor: '#fff'
    },
    previewSection: {
      marginTop: '20px',
      border: `1px solid ${customColors.primary}`,
      padding: '5px',
      backgroundColor: 'white',
      boxShadow: `0 0 10px rgba(0,0,0,0.1)`
    },
    etiquette: {
      border: `1px solid ${customColors.secondary}`,
      display: 'flex',
      flexDirection: 'row',
      minHeight: '240px'
    },
    logoAndGradesSection: {
      width: '33%',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '5px',
      borderRight: `1px solid ${customColors.secondary}`,
      backgroundColor: '#fff'
    },
    logoContainer: {
      marginTop: '10px',
      marginBottom: '15px',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center'
    },
    gradesContainer: {
      width: '100%',
      marginTop: '15px',
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      gap: '5px',
      padding: '0 10px'
    },
    infoSection: {
      width: '33%',
      padding: '5px',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      backgroundColor: '#fff'
    },
    noteSection: {
      width: '33%',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: pokemonTypes[formData.selectedType].background,
      padding: '5px',
      borderLeft: `1px solid ${customColors.secondary}`
    },
    productTitle: {
      fontWeight: 'bold',
      textAlign: 'center',
      fontSize: '20px',
      marginBottom: '8px'
    },
    productInfo: {
      textAlign: 'center',
      fontSize: '14px',
      margin: '3px 0',
      fontWeight: 'medium'
    },
    gradeLabel: {
      fontWeight: 'bold',
      fontSize: '14px'
    },
    gradeValue: {
      textAlign: 'right',
      fontSize: '14px'
    },
    globalNote: {
      fontSize: '48px',
      fontWeight: 'bold',
      color: customColors.primary
    },
    etat: {
      fontSize: '14px',
      fontWeight: 'bold',
      marginTop: '5px'
    },
    logoText: {
      fontWeight: 'bold',
      fontSize: '12px',
      marginTop: '10px',
      textAlign: 'center'
    },
    colorPicker: {
      display: 'flex',
      alignItems: 'center',
      marginBottom: '10px'
    },
    colorLabel: {
      width: '150px',
      fontWeight: 'bold'
    },
    typePreview: {
      display: 'flex',
      flexWrap: 'wrap',
      gap: '10px',
      marginTop: '15px'
    },
    typeChip: {
      padding: '5px 10px',
      borderRadius: '15px',
      display: 'inline-block',
      margin: '5px',
      cursor: 'pointer',
      fontWeight: 'bold',
      fontSize: '12px',
      transition: 'transform 0.2s',
      textAlign: 'center'
    }
  };

  // Gestion du téléchargement du logo
  const handleLogoUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.type.match('image.*')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedLogo(file);
        setLogoPreview(e.target.result);
      };
      reader.readAsDataURL(file);
      setFormData({...formData, logoOption: 'upload'});
    }
  };

  // Générer le logo selon l'option sélectionnée
  const renderLogo = () => {
    switch(formData.logoOption) {
      case 'default':
        // Logo SVG par défaut avec les couleurs du type
        return (
          <svg width="80" height="60" viewBox="0 0 100 80">
            <g transform={`translate(10,10) scale(0.8)`}>
              <polygon points="20,10 80,10 90,40 80,70 20,70 10,40" fill="white" stroke="black" strokeWidth="3"/>
              <polygon points="25,20 75,20 85,40 75,60 25,60 15,40" fill={customColors.primary} stroke="black" strokeWidth="2"/>
              <polygon points="30,30 70,30 75,40 70,50 30,50 25,40" fill={customColors.secondary} stroke="black" strokeWidth="2"/>
            </g>
          </svg>
        );
      
      case 'text':
        // Logo textuel simple
        return (
          <div style={{
            width: '80px',
            height: '60px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            border: '1px solid #ddd',
            borderRadius: '4px',
            backgroundColor: customColors.primary,
            color: '#fff',
            fontWeight: 'bold',
            fontSize: formData.logoText.length > 10 ? '12px' : '14px',
            textAlign: 'center',
            padding: '5px'
          }}>
            {formData.logoText}
          </div>
        );
      
      case 'upload':
        // Logo téléchargé par l'utilisateur
        return logoPreview ? (
          <img 
            src={logoPreview} 
            alt="Logo personnalisé" 
            style={{
              maxWidth: '80px', 
              maxHeight: '60px', 
              objectFit: 'contain'
            }} 
          />
        ) : (
          <div style={{
            width: '80px',
            height: '60px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            border: '1px dashed #999',
            backgroundColor: '#f0f0f0',
            fontSize: '12px',
            textAlign: 'center'
          }}>
            Aucun logo téléchargé
          </div>
        );
      
      default:
        return renderTypeLogo();
    }
  };

  // Générer le SVG avec les couleurs du type sélectionné
  const renderTypeLogo = () => {
    return (
      <svg width="80" height="60" viewBox="0 0 100 80">
        <g transform={`translate(10,10) scale(0.8)`}>
          <polygon points="20,10 80,10 90,40 80,70 20,70 10,40" fill="white" stroke="black" strokeWidth="3"/>
          <polygon points="25,20 75,20 85,40 75,60 25,60 15,40" fill={customColors.primary} stroke="black" strokeWidth="2"/>
          <polygon points="30,30 70,30 75,40 70,50 30,50 25,40" fill={customColors.secondary} stroke="black" strokeWidth="2"/>
        </g>
      </svg>
    );
  };

  // Générer des puces représentant les différents types
  const renderTypeChips = () => {
    return Object.keys(pokemonTypes).map(type => (
      <div 
        key={type}
        style={{
          ...styles.typeChip,
          backgroundColor: pokemonTypes[type].primary,
          color: pokemonTypes[type].text,
          border: `2px solid ${pokemonTypes[type].secondary}`,
          transform: formData.selectedType === type ? 'scale(1.1)' : 'scale(1)',
          boxShadow: formData.selectedType === type ? '0 0 5px rgba(0,0,0,0.3)' : 'none'
        }}
        onClick={() => {
          setFormData({...formData, selectedType: type});
          setCustomColors({
            primary: pokemonTypes[type].primary,
            secondary: pokemonTypes[type].secondary
          });
        }}
      >
        {type}
      </div>
    ));
  };

  return (
    <div style={styles.container}>
      <h1 style={{fontSize: '24px', fontWeight: 'bold', marginBottom: '20px'}}>
        Générateur d'Étiquettes Pokémon
      </h1>
      
      <div style={styles.formSection}>
        <h2 style={{fontSize: '18px', marginBottom: '15px'}}>Personnalisation du Logo</h2>
        
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '10px',
          marginBottom: '20px'
        }}>
          <div 
            style={{
              padding: '10px',
              border: formData.logoOption === 'default' ? `2px solid ${customColors.primary}` : '1px solid #ddd',
              borderRadius: '5px',
              cursor: 'pointer',
              backgroundColor: formData.logoOption === 'default' ? '#f0f0f0' : '#fff',
              textAlign: 'center',
              width: 'calc(33% - 10px)'
            }}
            onClick={() => setFormData({...formData, logoOption: 'default'})}
          >
            <div style={{marginBottom: '10px'}}>
              {renderTypeLogo()}
            </div>
            <div>Logo par défaut</div>
          </div>
          
          <div 
            style={{
              padding: '10px',
              border: formData.logoOption === 'text' ? `2px solid ${customColors.primary}` : '1px solid #ddd',
              borderRadius: '5px',
              cursor: 'pointer',
              backgroundColor: formData.logoOption === 'text' ? '#f0f0f0' : '#fff',
              textAlign: 'center',
              width: 'calc(33% - 10px)'
            }}
            onClick={() => setFormData({...formData, logoOption: 'text'})}
          >
            <div style={{
              marginBottom: '10px',
              width: '80px',
              height: '60px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: '1px solid #ddd',
              borderRadius: '4px',
              backgroundColor: customColors.primary,
              color: '#fff',
              fontWeight: 'bold',
              margin: '0 auto'
            }}>
              TEXTE
            </div>
            <div>Logo texte</div>
          </div>
          
          <div 
            style={{
              padding: '10px',
              border: formData.logoOption === 'upload' ? `2px solid ${customColors.primary}` : '1px solid #ddd',
              borderRadius: '5px',
              cursor: 'pointer',
              backgroundColor: formData.logoOption === 'upload' ? '#f0f0f0' : '#fff',
              textAlign: 'center',
              width: 'calc(33% - 10px)'
            }}
            onClick={() => document.getElementById('logo-upload').click()}
          >
            <div style={{marginBottom: '10px'}}>
              {logoPreview ? (
                <img 
                  src={logoPreview} 
                  alt="Logo téléchargé" 
                  style={{
                    maxWidth: '80px', 
                    maxHeight: '60px', 
                    objectFit: 'contain',
                    margin: '0 auto'
                  }} 
                />
              ) : (
                <div style={{
                  width: '80px',
                  height: '60px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: '1px dashed #999',
                  backgroundColor: '#f0f0f0',
                  fontSize: '12px',
                  margin: '0 auto'
                }}>
                  Télécharger
                </div>
              )}
            </div>
            <div>Logo personnalisé</div>
            <input 
              id="logo-upload"
              type="file" 
              accept="image/*" 
              onChange={handleLogoUpload}
              style={{display: 'none'}}
            />
          </div>
        </div>
        
        {formData.logoOption === 'text' && (
          <div style={styles.inputField}>
            <label style={styles.label}>Texte du logo</label>
            <input 
              type="text" 
              name="logoText" 
              value={formData.logoText} 
              onChange={handleChange}
              style={styles.input}
              maxLength="20"
            />
          </div>
        )}

        <h2 style={{fontSize: '18px', marginBottom: '15px', marginTop: '25px'}}>Type de Carte Pokémon</h2>
        
        <div style={styles.inputField}>
          <label style={styles.label}>Sélectionnez un type</label>
          <select 
            name="selectedType" 
            value={formData.selectedType} 
            onChange={handleChange}
            style={styles.select}
          >
            {Object.keys(pokemonTypes).map(type => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>
        </div>
        
        <div style={styles.typePreview}>
          {renderTypeChips()}
        </div>
        
        <h3 style={{fontSize: '16px', marginTop: '20px', marginBottom: '10px'}}>Personnalisation des couleurs</h3>
        
        <div style={styles.colorPicker}>
          <span style={styles.colorLabel}>Couleur principale:</span>
          <input 
            type="color" 
            name="primary"
            value={customColors.primary}
            onChange={handleColorChange}
          />
          <span style={{marginLeft: '10px'}}>{customColors.primary}</span>
        </div>
        
        <div style={styles.colorPicker}>
          <span style={styles.colorLabel}>Couleur secondaire:</span>
          <input 
            type="color" 
            name="secondary"
            value={customColors.secondary}
            onChange={handleColorChange}
          />
          <span style={{marginLeft: '10px'}}>{customColors.secondary}</span>
        </div>
        
        <h2 style={{fontSize: '18px', marginTop: '20px', marginBottom: '15px'}}>Informations de la Carte</h2>
        
        <div style={styles.inputGroup}>
          <div style={styles.inputField}>
            <label style={styles.label}>Nom du Pokémon</label>
            <input 
              type="text" 
              name="nomCarte" 
              value={formData.nomCarte} 
              onChange={handleChange}
              style={styles.input}
            />
          </div>
          
          <div style={styles.inputField}>
            <label style={styles.label}>Édition</label>
            <input 
              type="text" 
              name="edition" 
              value={formData.edition} 
              onChange={handleChange}
              style={styles.input}
            />
          </div>
          
          <div style={styles.inputField}>
            <label style={styles.label}>Série</label>
            <input 
              type="text" 
              name="serie" 
              value={formData.serie} 
              onChange={handleChange}
              style={styles.input}
            />
          </div>
          
          <div style={styles.inputField}>
            <label style={styles.label}>Année/Info</label>
            <input 
              type="text" 
              name="annee" 
              value={formData.annee} 
              onChange={handleChange}
              style={styles.input}
            />
          </div>
          
          <div style={styles.inputField}>
            <label style={styles.label}>Numéro de série</label>
            <input 
              type="text" 
              name="numeroSerie" 
              value={formData.numeroSerie} 
              onChange={handleChange}
              style={styles.input}
            />
          </div>
          
          <div style={styles.inputField}>
            <label style={styles.label}>Note Globale</label>
            <input 
              type="text" 
              name="noteGlobale" 
              value={formData.noteGlobale} 
              onChange={handleChange}
              style={styles.input}
            />
          </div>
          
          <div style={styles.inputField}>
            <label style={styles.label}>Centrage</label>
            <input 
              type="text" 
              name="noteCentrage" 
              value={formData.noteCentrage} 
              onChange={handleChange}
              style={styles.input}
            />
          </div>
          
          <div style={styles.inputField}>
            <label style={styles.label}>Coins</label>
            <input 
              type="text" 
              name="noteCoins" 
              value={formData.noteCoins} 
              onChange={handleChange}
              style={styles.input}
            />
          </div>
          
          <div style={styles.inputField}>
            <label style={styles.label}>Côtés</label>
            <input 
              type="text" 
              name="noteCotes" 
              value={formData.noteCotes} 
              onChange={handleChange}
              style={styles.input}
            />
          </div>
          
          <div style={styles.inputField}>
            <label style={styles.label}>Surface</label>
            <input 
              type="text" 
              name="noteSurface" 
              value={formData.noteSurface} 
              onChange={handleChange}
              style={styles.input}
            />
          </div>
          
          <div style={styles.inputField}>
            <label style={styles.label}>État</label>
            <input 
              type="text" 
              name="etatLabel" 
              value={formData.etatLabel} 
              onChange={handleChange}
              style={styles.input}
            />
          </div>
          
          <div style={styles.inputField}>
            <label style={styles.label}>Texte du logo</label>
            <input 
              type="text" 
              name="logoText" 
              value={formData.logoText} 
              onChange={handleChange}
              style={styles.input}
            />
          </div>
        </div>
      </div>
      
      <h2 style={{fontSize: '18px', marginBottom: '10px', marginTop: '20px'}}>Aperçu de l'Étiquette</h2>
      
      <div style={styles.previewSection}>
        <div style={styles.etiquette}>
          {/* Section Logo et notes d'évaluation */}
          <div style={styles.logoAndGradesSection}>
            <div style={styles.logoContainer}>
              {renderLogo()}
              <div style={styles.logoText}>{formData.logoText}</div>
            </div>
            
            <div style={styles.gradesContainer}>
              <div style={styles.gradeLabel}>CENTRAGE</div>
              <div style={styles.gradeValue}>{formData.noteCentrage}</div>
              
              <div style={styles.gradeLabel}>CÔTÉS</div>
              <div style={styles.gradeValue}>{formData.noteCotes}</div>
              
              <div style={styles.gradeLabel}>COINS</div>
              <div style={styles.gradeValue}>{formData.noteCoins}</div>
              
              <div style={styles.gradeLabel}>SURFACE</div>
              <div style={styles.gradeValue}>{formData.noteSurface}</div>
            </div>
          </div>
          
          {/* Section Info */}
          <div style={styles.infoSection}>
            <div>
              <div style={styles.productTitle}>{formData.nomCarte}</div>
              <div style={styles.productInfo}>{formData.edition}</div>
              <div style={styles.productInfo}>{formData.serie}</div>
              <div style={styles.productInfo}>{formData.annee}</div>
              <div style={styles.productInfo}>{formData.numeroSerie}</div>
            </div>
          </div>
          
          {/* Section Note */}
          <div style={styles.noteSection}>
            <div style={styles.globalNote}>{formData.noteGlobale}</div>
            <div style={styles.etat}>{formData.etatLabel}</div>
          </div>
        </div>
      </div>
      
      <div style={{
        marginTop: '20px', 
        padding: '15px', 
        backgroundColor: pokemonTypes[formData.selectedType].background, 
        borderRadius: '5px',
        border: `1px solid ${customColors.secondary}`
      }}>
        <h3 style={{fontWeight: 'bold', marginBottom: '10px', color: customColors.primary}}>
          Instructions:
        </h3>
        <ol style={{paddingLeft: '20px'}}>
          <li style={{marginBottom: '5px'}}>Choisissez le type de Pokémon correspondant à votre carte</li>
          <li style={{marginBottom: '5px'}}>Ajustez les couleurs si nécessaire</li>
          <li style={{marginBottom: '5px'}}>Remplissez les informations spécifiques à votre carte</li>
          <li style={{marginBottom: '5px'}}>Utilisez une capture d'écran pour enregistrer l'étiquette</li>
        </ol>
        <p style={{marginTop: '10px', fontSize: '12px', fontStyle: 'italic'}}>
          Note: Dans une version plus avancée, cette application pourrait extraire automatiquement les couleurs dominantes de l'image de votre carte Pokémon.
        </p>
      </div>
    </div>
  );
};

export default PokemonCardEtiquetteGenerator;
