#ifndef DEVICEWIDGET_H
#define DEVICEWIDGET_H

#include "UseDevice.h"

#include "ui_DeviceWidget.h"
#include <QWidget>
#include <QTimer>



namespace Ui {
class DeviceWidget;
}

class DeviceWidget : public QWidget
{
    Q_OBJECT

public:
    
    explicit DeviceWidget(QWidget *parent = nullptr);
    ~DeviceWidget();
    void closeEvent(QCloseEvent*);
    
    
signals:
    void ExitWin();

private:
    Ui::DeviceWidget ui;
    QTimer* timer;

    HDErrorInfo error;
    HHD hHD;
    HDCallbackCode InfoCallback;
    DeviceInfo DeviceWidgetInfo;

private slots:
    void showData();

};

#endif // DEVICEWIDGET_H
