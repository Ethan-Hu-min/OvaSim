#ifndef  GLWIDGET_H
#define GLWIDGET_H

#include  "USRenderer.h"

#include <QOpenGLWidget>
#include<QOpenGLExtraFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>

#include<QTime>




class GLWidget : public QOpenGLWidget, protected QOpenGLExtraFunctions {

	Q_OBJECT

public: 
	GLWidget(QWidget* parent = nullptr);
	~GLWidget();

	void createRenderer();


public slots:
	void initializeGL() Q_DECL_OVERRIDE;
	void resizeGL(int w, int h) Q_DECL_OVERRIDE;
	void paintGL() Q_DECL_OVERRIDE;
	void setImage();
	//void initTextures();
	//void initShaders();

private:
	USRenderer* usRenderer = nullptr;
	int frameSizeWidth = 0;
	int frameSizeHeight = 0;
	int frameNo = 0;
	int lastTime;
	int fps = 0;
	//QVector<QVector3D> vertices;
	//QVector<QVector2D> texCoords;
	//QOpenGLShaderProgram program;
	//QOpenGLTexture* texture = nullptr;

	GLuint displayTexture; 

	 

};




#endif









