#ifndef  GLWIDGET_H
#define GLWIDGET_H

#include  "USRenderer.h"

#include <QOpenGLWidget>
#include<QOpenGLExtraFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>

#include<QTime>
#include <qstring.h>

#include "UseDevice.h"

class GLWidget : public QOpenGLWidget, protected QOpenGLExtraFunctions {

	Q_OBJECT

signals: 	void fpsChanged(int fps);
signals:	void needleSwitchSignal(QString status);
signals:	void ovamNumsSignal(int nums);

public: 
	GLWidget(QWidget* parent = nullptr);
	~GLWidget();

	void createRenderer();
	void keyPressEvent(QKeyEvent* event);


public slots:
	void initializeGL() Q_DECL_OVERRIDE;
	void resizeGL(int w, int h) Q_DECL_OVERRIDE;
	void paintGL() Q_DECL_OVERRIDE;
	void setImage();

	void setStartRenderTrue();
	void setStartRenderFalse();

	void setOriginDevice();
	//void initTextures();
	//void initShaders();

private:
	USRenderer* usRenderer = nullptr;
	int frameSizeWidth = 0;
	int frameSizeHeight = 0;
	int frameNo = 0;
	int lastTime;
	int fps = 0;

	bool startRender = false;
	int controlMode = 1;//1 for keyboard; 2 for device
	bool needleSwitch = false;

	vec3f originTransducerPos;
	vec3f originTransducerDir;
	vec3f originTransducerVer;

	float originTransducerAngle;

	vec3f originDevicePos;
	vec3f originDeviceAngles;
	//QVector<QVector3D> vertices;
	//QVector<QVector2D> texCoords;
	//QOpenGLShaderProgram program;
	//QOpenGLTexture* texture = nullptr;

	GLuint displayTexture; 

	HDErrorInfo error;
	HHD hHD;
	HDSchedulerHandle hForceGLCallback;
	DeviceInfo DeviceWidgetInfo;

};




#endif









